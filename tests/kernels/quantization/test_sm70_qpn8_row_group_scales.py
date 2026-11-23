# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QPN8 decode kernel with MXFP8 row-group scales on the GLM KDA o_proj shape
(N=4096, K=8192, M=1..8; split-K 16 -> 16 scale groups per warp).

The kernel loads each scale one scale group ahead; the values and the math are
unchanged. Bit equality against the pre-change kernel is checked across two
builds by ``pkg_qpn8_scales_bench_2026-11-23.py --dump / --check``; here the
output is held against the dequantized weight and against itself.
"""

import pytest
import torch
from torch import nn

from vllm.model_executor.kernels.linear.qpn import pre_ampere  # noqa: F401
from vllm.model_executor.layers.quantization import sm70_turbomind as tm
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    dequant_mxfp8_to_bf16,
)

KDA_O_PROJ_N = 4096
KDA_O_PROJ_K = 8192


def _require_sm7x():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (7, 0),
        (7, 5),
    ):
        pytest.skip("requires SM70 or SM75")
    import vllm._C  # noqa: F401


def _mxfp8_layer(n: int, k: int):
    low, high = tm.MXFP8_QPN8_EXPONENT_RANGE
    weight = (torch.randn(n, k, device="cuda") * 8).to(torch.float8_e4m3fn)
    scale = torch.randint(
        low, high + 1, (n, k // MXFP8_BLOCK_SIZE), device="cuda", dtype=torch.uint8
    )
    layer = nn.Module()
    tm.prepare_mxfp8_qpn8_linear(layer, weight, scale)
    return layer, dequant_mxfp8_to_bf16(weight, scale).float()


@pytest.mark.parametrize("rows", range(1, 9))
@pytest.mark.parametrize(
    "n,k", [(KDA_O_PROJ_N, KDA_O_PROJ_K), (4096, 4096), (2048, 6144)]
)
def test_row_group_scale_decode_matches_reference_and_is_repeatable(rows, n, k):
    _require_sm7x()
    torch.manual_seed(3011)
    layer, reference_weight = _mxfp8_layer(n, k)
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.1

    first = tm.apply_prepared_fp8_qpn8_linear(layer, x, None)
    second = tm.apply_prepared_fp8_qpn8_linear(layer, x, None)
    reference = x.float() @ reference_weight.t()

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert (first.float() - reference).abs().max() <= 2e-3 * reference.abs().max()
