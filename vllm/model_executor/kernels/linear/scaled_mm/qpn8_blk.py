# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Block-scaled FP8 linears on SM70/SM75 through 1Cat's native QPN8 (fork addition).

Serves blockwise-FP8 linears (weight_block_size [128, 128], DeepSeek-class
attention and shared experts) on Volta and Turing with the fp8_qpn8_* operators
of csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu. 1Cat admits those operators only
for measured shape tables inside its exact-SM70 TurboMind branch; this kernel
admits every block-FP8 linear whose shape the operators accept, on both card
generations (the sm_70 cubins run on sm_75, and TurboMind's FP8 GEMM has no
sm_75 kernel at all).

Weight-only: activations stay fp16 (apply_input_quant=False). M <= 8 runs the
QPN8 GEMM on the packed codes; larger M dequantizes into a dense fp16 buffer
and runs cuBLAS, as 1Cat's own dispatch does for block scales. Each call takes
its buffer from the caching allocator on its own stream: DeepSeek-V4 runs the
indexer's wq_b on an aux stream next to the main wq_b, and one shared buffer
lets the two dequantized weights overwrite each other.
"""

import torch
from torch.library import custom_op

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)

# The QPN8 GEMM serves M up to this bound; larger M takes the dense prefill.
_QPN8_MAX_M = 8


@custom_op("sm70_fp8::qpn8_native_linear", mutates_args=())
def _qpn8_native_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    prefetch_codes: bool,
) -> torch.Tensor:
    # The M decision stays inside the opaque op: a compiled graph covers a
    # dynamic M range, so a Python branch traced at small M would be reused
    # for prefill.
    k_dim, n_dim = codes.shape
    out = x.new_empty((x.shape[0], n_dim))
    if x.shape[0] <= _QPN8_MAX_M:
        sm70_ops.fp8_qpn8_gemm_sm70_out(
            out,
            x,
            codes,
            group_scales,
            split_k,
            accumulator_chains,
            True,
            prefetch_codes,
        )
        return out
    dense = torch.empty((k_dim * n_dim,), dtype=torch.float16, device=x.device)
    sm70_ops.fp8_qpn8_prefill_sm70_out(
        out, dense.data_ptr(), x, codes, group_scales, False
    )
    return out


@_qpn8_native_linear.register_fake
def _qpn8_native_linear_fake(
    x, codes, group_scales, split_k, accumulator_chains, prefetch_codes
):
    return x.new_empty((x.shape[0], codes.shape[1]))


class QPN8Fp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """Block-scaled FP8 on SM70/SM75 via 1Cat's native QPN8 operators."""

    # fp16 activations go straight into the GEMM; no input quantization.
    apply_input_quant = False

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not torch.cuda.is_available():
            return False, "CUDA unavailable"
        from vllm.platforms import current_platform

        # The worker's own device decides: stages of a mixed pipeline differ.
        cap = current_platform.get_device_capability()
        if cap is None or tuple(cap) not in ((7, 0), (7, 5)):
            return False, f"{cap} is not sm70/sm75"
        return True, None

    @classmethod
    def can_implement(cls, config):
        if config.input_dtype != torch.float16:
            return False, "native QPN8 needs fp16 activations"
        n, k = config.weight_shape
        block = tuple(config.weight_quant_key.scale.group_shape)
        if block != (128, 128) or n % 128 or k % 128:
            return False, f"native QPN8 needs block [128,128], N,K % 128 ({n},{k})"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        from vllm.model_executor.layers.quantization.fp8 import _sm70_fp8_qpn8_config

        params = self._get_layer_params(layer)
        weight = params.weight.data
        n, k = weight.shape
        scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        assert scale is not None, "block-FP8 layer without a weight scale"
        block_scales = scale.data.detach().float().contiguous()
        if tuple(block_scales.shape) != (n // 128, k // 128):
            raise ValueError(
                f"QPN8: scale raster {tuple(block_scales.shape)} does not match "
                f"weights {n}x{k} at block [128,128]"
            )
        if weight.dtype != torch.float8_e4m3fn:
            weight = weight.view(torch.float8_e4m3fn)

        codes, group_scales = sm70_ops.fp8_qpn8_prepare_sm70(weight, block_scales)
        k_dim, n_dim = (int(dim) for dim in codes.shape)
        layer._qpn8_codes = codes
        layer._qpn8_scales = group_scales
        layer._qpn8_out_features = n
        layer._qpn8_cfg = _sm70_fp8_qpn8_config(k_dim, n_dim, False)
        # The packed codes are now the only resident copy.
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=torch.uint8, device=codes.device), requires_grad=False
        )

    def apply_block_scaled_mm(self, A, B, As, Bs):
        # Satisfies the ABC; apply_weights below bypasses the base-class
        # A/As machinery entirely (weight-only path, fp16 activations).
        raise RuntimeError("unreachable: QPN8 overrides apply_weights")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        codes = layer._qpn8_codes
        split_k, accumulator_chains, prefetch_codes = layer._qpn8_cfg
        y = torch.ops.sm70_fp8.qpn8_native_linear(
            x.reshape(-1, codes.shape[0]).contiguous(),
            codes,
            layer._qpn8_scales,
            split_k,
            accumulator_chains,
            prefetch_codes,
        )
        if bias is not None:
            y = y + bias
        return y.reshape(x.shape[:-1] + (layer._qpn8_out_features,))
