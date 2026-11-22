# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA glue kernels must equal the eager op sequence they replace, bit for bit.

Shapes are GLM-5.3-Flash's: linear_attn_config num_heads 64, head_dim 128
(TP1), rows 1/5/6/8 as verification and decode steps produce them.
"""

import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fla.ops.kda import FusedRMSNormGated
from vllm.model_executor.layers.fla.ops.sm70.kda_glue import (
    kda_gated_rmsnorm,
    kda_pack_recurrent_inputs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_HEADS = 64
HEAD_DIM = 128
PROJECTION = NUM_HEADS * HEAD_DIM
BETA_ROW = NUM_HEADS + 2 * HEAD_DIM  # in_proj_bfg_a row: b | f_a | g_a
ROWS = (1, 5, 6, 8)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("scale", [1.0, 1000.0, 1e-3])
def test_gated_rmsnorm_equals_eager_forward_native(rows: int, scale: float) -> None:
    config = VllmConfig(
        compilation_config=CompilationConfig(
            custom_ops=["none", "-fused_rms_norm_gated"]
        )
    )
    with set_current_vllm_config(config):
        norm = FusedRMSNormGated(
            HEAD_DIM, activation="sigmoid", device="cuda", dtype=torch.float16
        )
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))
    generator = torch.Generator(device="cuda").manual_seed(rows)
    x = (
        torch.randn((1, rows, NUM_HEADS, HEAD_DIM), device="cuda", generator=generator)
        * scale
    )
    gate = torch.randn(
        (rows, NUM_HEADS, HEAD_DIM),
        device="cuda",
        generator=generator,
    ).half()
    reference = norm(x, gate, out_dtype=torch.float16)
    result = kda_gated_rmsnorm(x, gate, norm.weight, norm.eps, out_dtype=torch.float16)
    assert result.shape == reference.shape and result.dtype == reference.dtype
    assert torch.equal(result, reference)


@pytest.mark.parametrize("rows", ROWS)
def test_pack_recurrent_inputs_equals_contiguous_copies(rows: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(rows)
    qkv = torch.randn((rows, 3 * PROJECTION), device="cuda", generator=generator).half()
    projected = torch.randn((rows, BETA_ROW), device="cuda", generator=generator).half()
    beta = projected[:, :NUM_HEADS]  # strided split, as in _project_inputs
    packed = kda_pack_recurrent_inputs(qkv, beta, NUM_HEADS, HEAD_DIM)
    q, k, v = (
        part.reshape(1, -1, NUM_HEADS, HEAD_DIM)
        for part in qkv.split(PROJECTION, dim=-1)
    )
    expected = (q.contiguous(), k.contiguous(), v.contiguous())
    for result, reference in zip(packed[:3], expected, strict=True):
        assert result.is_contiguous() and torch.equal(result, reference)
    assert packed[3].is_contiguous()
    assert torch.equal(packed[3], beta.unsqueeze(0).contiguous())
