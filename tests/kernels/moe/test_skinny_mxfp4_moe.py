# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 experts on the skinny FP4 kernels (SM70/SM75)."""

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.experts.nvfp4_skinny_moe import (
    Mxfp4SkinnySm70Experts,
    Nvfp4SkinnySm70Experts,
    rebase_e8m0_for_fp16,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)


def _pow2(e8m0: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, e8m0.to(torch.float64) - 127)


def test_rebase_reproduces_every_scale_exactly():
    generator = torch.Generator().manual_seed(0)
    low = torch.randint(95, 108, (4, 1, 1), generator=generator)
    offsets = torch.randint(0, 20, (4, 8, 6), generator=generator)
    scales = (low + offsets).to(torch.uint8)

    rebased = scales.clone()
    global_scale = rebase_e8m0_for_fp16(rebased)

    assert int(rebased.min()) >= 113 and int(rebased.max()) <= 142
    restored = _pow2(rebased) * global_scale.to(torch.float64).view(-1, 1, 1)
    assert torch.equal(restored, _pow2(scales))


def test_rebase_keeps_the_global_scale_inside_fp16():
    # All scales high: lifting the smallest to 113 would need a negative
    # shift whose global scale times 2^14 overflows fp16.
    scales = torch.full((1, 4, 4), 124, dtype=torch.uint8)
    scales[0, 0, 0] = 128

    rebased = scales.clone()
    global_scale = rebase_e8m0_for_fp16(rebased)

    assert float(global_scale) * 2**14 <= 2**15
    restored = _pow2(rebased) * float(global_scale)
    assert torch.equal(restored, _pow2(scales))


@pytest.mark.parametrize(
    ("low", "high", "message"),
    [
        (100, 255, "NaN"),
        (95, 125, "span"),
        (80, 90, "outside"),
    ],
)
def test_rebase_refuses_scales_the_kernels_cannot_represent(low, high, message):
    scales = torch.full((1, 2, 2), low, dtype=torch.uint8)
    scales[0, 1, 1] = high

    untouched = scales.clone()

    with pytest.raises(ValueError, match=message):
        rebase_e8m0_for_fp16(scales)
    assert torch.equal(scales, untouched)


def _experts_with_rasters(cls, w1_scales, w2_scales):
    experts = cls.__new__(cls)
    experts._w1_block_scales = w1_scales
    experts._w2_block_scales = w2_scales
    return experts


def test_nvfp4_rasters_keep_16_or_fold_to_32():
    # fp8-e4m3 bytes with zero mantissa are powers of two; duplicated pairs
    # are an MXFP4 raster in NVFP4 form.
    duplicated = torch.tensor([[[0x38, 0x38, 0x40, 0x40]]], dtype=torch.uint8)
    distinct = torch.tensor([[[0x38, 0x40, 0x40, 0x48]]], dtype=torch.uint8)
    layer = nn.Module()

    kept = _experts_with_rasters(Nvfp4SkinnySm70Experts, distinct, distinct.clone())
    folded = _experts_with_rasters(
        Nvfp4SkinnySm70Experts, duplicated, duplicated.clone()
    )

    assert kept._scale_rasters(layer)[2] == 16
    s13, _, group = folded._scale_rasters(layer)
    assert group == 32
    assert s13.tolist() == [[[0x38 // 8 + 120, 0x40 // 8 + 120]]]


def test_mxfp4_rasters_pass_through_at_32():
    scales = torch.full((2, 4, 2), 120, dtype=torch.uint8)
    experts = _experts_with_rasters(Mxfp4SkinnySm70Experts, scales, scales.clone())

    s13, s2, group = experts._scale_rasters(nn.Module())

    assert group == 32
    assert s13 is not None and torch.equal(s13, scales)
    assert torch.equal(s2, scales)


def _skinny_devices() -> list[int]:
    if not torch.cuda.is_available():
        return []
    return [
        i
        for i in range(torch.accelerator.device_count())
        if torch.cuda.get_device_capability(i) in ((7, 0), (7, 5))
    ]


@pytest.mark.skipif(not _skinny_devices(), reason="needs an SM70 or SM75 GPU")
@pytest.mark.parametrize("capability", [(7, 0), (7, 5)])
def test_moe_qpn_mxfp4_mode_matches_the_checkpoint_scales(capability):
    devices = [
        i
        for i in _skinny_devices()
        if torch.cuda.get_device_capability(i) == capability
    ]
    if not devices:
        pytest.skip(f"no GPU with capability {capability}")
    from vllm.model_executor.kernels.linear.nvfp4.marlin import (
        _get_skinny_ext,
        _qpn_prepack,
    )

    device = torch.device("cuda", devices[0])
    torch.accelerator.set_device_index(device.index)
    generator = torch.Generator().manual_seed(1)
    experts, rows, cols, tokens = 2, 64, 256, 5
    codes = torch.randint(0, 256, (experts, rows, cols // 2), generator=generator)
    codes = codes.to(torch.uint8)
    scales = (
        torch.randint(0, 8, (experts, rows, cols // 32), generator=generator) + 112
    ).to(torch.uint8)
    hidden = torch.randn(tokens, cols, generator=generator).to(torch.float16)

    reference = []
    for e in range(experts):
        values = break_fp4_bytes(codes[e], torch.float32)
        weight = values * _pow2(scales[e]).repeat_interleave(32, dim=1).float()
        reference.append(hidden.float() @ weight.T)

    rebased = scales.clone()
    global_scale = rebase_e8m0_for_fp16(rebased)
    w = codes.to(device)
    s = rebased.to(device)
    for e in range(experts):
        qc, qs = _qpn_prepack(w[e], s[e], 32)
        w[e].view(-1).copy_(qc)
        s[e].view(-1).copy_(qs)

    ext = _get_skinny_ext()
    x = hidden.to(device)
    for e in range(experts):
        perm = torch.arange(tokens, dtype=torch.int32, device=device)
        gids = torch.full((tokens,), e, dtype=torch.int32, device=device)
        goff = torch.full((tokens + 1,), tokens, dtype=torch.int32, device=device)
        goff[0] = 0
        out = torch.empty((tokens, rows), dtype=torch.float16, device=device)
        ext.moe_qpn(
            x,
            w,
            s,
            global_scale.to(device),
            perm,
            gids,
            goff,
            1,
            out,
            False,
            tokens,
            16,
            1,
            1,
        )
        torch.testing.assert_close(
            out.float().cpu(), reference[e], rtol=2e-2, atol=2e-2
        )
