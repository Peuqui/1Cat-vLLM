# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NVFP4 quantization emulation for MoE.

This file implements NVFP4 emulation for NVFP4 MOE in case the hardware used does not
natively support NVFP4 MOE.

Weights are dequantized on the fly during each forward, we fall back to calling
`TritonExperts` using BF16, and fake NVFP4 quantize-dequantize
is applied on `a13`, `a2`.

Fork addition (v100-skinny): the dequantization is chunked over the experts
that the router actually selected. Upstream materializes the COMPLETE expert
stack of a layer in fp16 (DeepSeek-V4-Flash: 256 experts, ~13 GB per layer,
plus int64/fp32 intermediates inside ``break_fp4_bytes``), which is only
viable for the small MoEs this emulation was written for. Decode touches
``num_tokens * top_k`` experts at most -- 6 of 256 for a single token -- so
dequantizing only those, a chunk at a time, turns an OOM into a working
(slow) reference path. Each chunk runs as its own expert-parallel shard: the
chunk's experts are the local ones, everything else is masked out via
``expert_map`` (the Triton kernel writes zeros for ``expert_id == -1``), and
the per-chunk partial sums are accumulated.
"""

import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)

logger = init_logger(__name__)

# Experts dequantized per pass. The transient peak is dominated by the int64
# lookup indices inside break_fp4_bytes (8 bytes per weight element), so a
# chunk of 4 costs roughly 1.7 GB on DeepSeek-V4-Flash geometry -- small
# enough to survive next to a KV cache sized at gpu_memory_utilization.
_EMULATION_CHUNK = int(os.environ.get("VLLM_SM70_NVFP4_EMU_CHUNK", "4"))


class Nvfp4QuantizationEmulationTritonExperts(TritonExperts):
    """
    Extension of TritonExperts to support emulated NVFP4 MoE experts.

    It may be used for NVFP4 models when the device does not have
    native support for this dtype.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        logger.warning_once(
            "Using Nvfp4QuantizationEmulationTritonExperts MOE backend. This will"
            " dequantize weights on the fly and may be slower than native"
            " quantized MOE. Consider using a device with native quantization"
            " support (e.g. Nvidia Blackwell) for better performance."
        )
        logger.warning_once(
            "NVFP4 emulation dequantizes %d router-selected experts per pass"
            " (VLLM_SM70_NVFP4_EMU_CHUNK).",
            _EMULATION_CHUNK,
        )

        # `TritonExperts.apply` expects pre-dequantized weights,
        # which we handle in `apply` below.
        self.w1_scale_val = self.quant_config.w1_scale
        self.w2_scale_val = self.quant_config.w2_scale

        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

        self.quantization_emulation = True

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return "nvfp4"

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    def _dequantize_experts(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        global_scales: torch.Tensor,
        expert_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize the given experts of one packed weight stack."""
        return dequantize_to_dtype(
            tensor_fp4=packed[expert_ids],
            tensor_sf=scales[expert_ids],
            global_scale=global_scales[expert_ids],
            dtype=dtype,
            block_size=16,
            swizzle=False,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        """
        Apply emulated quantized MoE computation.

        This dequantizes the router-selected weights on the fly, in chunks of
        experts, and calls fused_experts_impl once per chunk.
        """
        # For NVFP4, weights are packed in uint8 format
        # w1 shape: [num_experts, 2*intermediate_size, hidden_size//2]
        # w2 shape: [num_experts, hidden_size, intermediate_size//2]
        assert w1.dtype == torch.uint8
        assert w2.dtype == torch.uint8

        if expert_map is not None:
            # The chunk loop below owns the expert_map slot; composing it with
            # an expert-parallel map would need the two mappings folded into
            # one, which no current configuration asks for.
            raise NotImplementedError(
                "Chunked NVFP4 emulation does not support expert parallelism."
            )

        if global_num_experts == -1:
            global_num_experts = w1.size(0)

        compute_dtype = hidden_states.dtype

        hidden_states, _ = moe_kernel_quantize_input(
            A=hidden_states,
            A_scale=self.quant_config.a1_gscale,
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            quantization_emulation=True,
        )

        # Only the experts the router picked need to exist in fp16. Their ids
        # index the full stack; the chunk-local ids are 0..len(chunk)-1.
        selected_experts = torch.unique(topk_ids.flatten()).to(torch.long)
        assert int(selected_experts[0]) >= 0, "topk_ids carries an invalid expert"
        num_selected = selected_experts.numel()

        accumulator: torch.Tensor | None = None
        for start in range(0, num_selected, _EMULATION_CHUNK):
            chunk_ids = selected_experts[start : start + _EMULATION_CHUNK]

            w1_dequant = self._dequantize_experts(
                w1, self.w1_scale_val, self.quant_config.g1_alphas,
                chunk_ids, compute_dtype,
            )
            w2_dequant = self._dequantize_experts(
                w2, self.w2_scale_val, self.quant_config.g2_alphas,
                chunk_ids, compute_dtype,
            )

            # Chunk-as-expert-parallel-shard: experts outside the chunk map to
            # -1 and the Triton kernel writes zeros for their blocks, so the
            # per-chunk result is the partial sum over this chunk's experts.
            chunk_map = torch.full(
                (global_num_experts,), -1, dtype=torch.int32, device=topk_ids.device
            )
            chunk_map[chunk_ids] = torch.arange(
                chunk_ids.numel(), dtype=torch.int32, device=topk_ids.device
            )

            # Activation quantization/dequantization is deferred to
            # `moe_kernel_quantize_input` in TritonExperts.apply.
            super().apply(
                output=output,
                hidden_states=hidden_states,
                w1=w1_dequant,
                w2=w2_dequant,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=chunk_map,
                a1q_scale=None,
                a2_scale=self.quant_config.a2_gscale,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            del w1_dequant, w2_dequant

            if num_selected <= _EMULATION_CHUNK:
                return

            # `output` may alias workspace13, which the next chunk overwrites,
            # so the partial sum is carried in a buffer of its own.
            if accumulator is None:
                accumulator = output.clone()
            else:
                accumulator.add_(output)

        assert accumulator is not None
        output.copy_(accumulator)
