# SPDX-License-Identifier: Apache-2.0
# v100-skinny, 2026. Dequantize the QPN fragment-order NVFP4 prepack back to a
# dense fp16 [N, K] weight for large-M (prefill) GEMMs via cuBLAS.
#
# Why: the sm75 skinny path used to keep three weight layouts per linear layer
# (checkpoint-native, QPN prepack, marlin repack), ~10 GiB each on Qwen3.8-27B.
# With this kernel the QPN prepack is the only resident layout: decode keeps
# the QPN kernels, prefill dequantizes one layer at a time into a transient
# fp16 buffer and runs torch.matmul. Marlin's FP4 path on Turing reaches only
# ~27 TFLOPS; cuBLAS fp16 tensor cores do better, so this is a memory AND a
# speed change.
#
# Layout (see marlin.py::_qpn_prepack): codes are [tiles=N/32][groups=K/16]
# [lane=32][8 bytes]; the lane's 8 bytes hold 16 nibbles for the 16 k of the
# group in korder = (0,2,4,6,1,3,5,7, 8,10,12,14,9,11,13,15); byte b holds
# korder[2b] in its low nibble and korder[2b+1] in its high nibble. The lane
# owns row n = tile*32 + col(lane) with
# col = ((lane>>2)&3)*8 + (lane&3) + ((lane&16)>0)*4. scales are
# [tiles][groups][lane] fp8-e4m3, one per (n, group).
# value = e2m1(code) * fp8(scale) * gscale, computed in fp32, stored as fp16 —
# the same product the QPN kernels form (scale*gscale in the epilogue).

import torch
import triton
import triton.language as tl

# e2m1 magnitudes indexed by the low three code bits; bit 3 is the sign.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_KORDER = (0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15)


def _lane_to_col() -> torch.Tensor:
    lane = torch.arange(32)
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4


def qpn_dequant_reference(
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    gscale: float,
    n: int,
    k: int,
) -> torch.Tensor:
    """Pure-torch inverse of _qpn_prepack (slow, for tests and as the spec)."""
    tiles, groups = n // 32, k // 16
    dev = qpn_codes.device
    qc = qpn_codes.view(tiles, groups, 32, 8)
    qs = qpn_scales.view(tiles, groups, 32)
    nib = torch.stack([qc & 0xF, qc >> 4], dim=-1).view(tiles, groups, 32, 16)
    korder = torch.tensor(_KORDER, device=dev)
    # nib[..., j] belongs to k = group*16 + korder[j]; invert to natural k order.
    inv = torch.empty(16, dtype=torch.long, device=dev)
    inv[korder] = torch.arange(16, device=dev)
    nib = nib[..., inv]  # [tiles, groups, lane, 16] in k order
    mags = torch.tensor(_E2M1_MAGNITUDES, device=dev, dtype=torch.float32)
    vals = mags[(nib & 7).long()] * torch.where(nib & 8 > 0, -1.0, 1.0)
    scale = qs.view(torch.float8_e4m3fn).to(torch.float32)  # [tiles, groups, lane]
    vals = vals * scale.unsqueeze(-1) * gscale
    # rows: n = tile*32 + col(lane)
    col = _lane_to_col().to(dev)
    out = torch.empty(n, k, dtype=torch.float32, device=dev)
    rows = (torch.arange(tiles, device=dev).view(tiles, 1) * 32 + col.view(1, 32))
    # vals -> [tiles, lane, groups, 16] -> [tiles, lane, k]
    out[rows.view(-1)] = vals.permute(0, 2, 1, 3).reshape(tiles * 32, k)
    return out.to(torch.float16)


@triton.jit
def _e2m1_value(code):
    """e2m1 nibble -> float: magnitudes 0,.5,1,1.5,2,3,4,6; bit 3 is the sign."""
    mag = code & 7
    m_f = mag.to(tl.float32)
    val = tl.where(mag < 4, m_f * 0.5, tl.where(mag < 6, m_f - 2.0, tl.where(mag == 6, 4.0, 6.0)))
    return tl.where((code & 8) > 0, -val, val)


@triton.jit
def _e4m3_value(b):
    """fp8-e4m3fn byte -> float, arithmetic (Volta/Turing have no fp8 conversions).

    bias 7; exponent 0 is subnormal (mantissa/8 * 2^-6); the NaN code 0x7f
    never occurs in ModelOpt scales.
    """
    sign = tl.where((b & 0x80) > 0, -1.0, 1.0)
    exp = ((b >> 3) & 0xF).to(tl.float32)
    mant = (b & 7).to(tl.float32) / 8.0
    normal = (1.0 + mant) * tl.exp2(exp - 7.0)
    subnormal = mant * tl.exp2(-6.0)
    return sign * tl.where(exp == 0, subnormal, normal)


@triton.jit
def _qpn_dequant_kernel(
    codes32_ptr,
    scales_ptr,
    out_ptr,
    gscale,
    groups,
    K,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    # One program: one tile (32 rows) x GROUPS_PER_BLOCK groups. A lane's
    # 8-byte payload is read as two 32-bit words (64-bit shifts are slow on
    # Volta/Turing); nibble j sits at bit 4*(j&7) of word j>>3 and holds k
    # offset korder[j]. Scales are read once per (lane, group).
    tile = tl.program_id(0)
    gblock = tl.program_id(1)
    lane = tl.arange(0, 32)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).to(tl.int32) * 4
    row = tile * 32 + col  # [32]
    g = gblock * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)  # [G]
    valid = g < groups
    lane_base = (tile * groups + g) * 32  # [G] first lane of (tile, g)
    idx = lane_base[None, :] + lane[:, None]  # [32, G] lane index
    w0 = tl.load(codes32_ptr + idx * 2, mask=valid[None, :], other=0)  # nibbles 0..7
    w1 = tl.load(codes32_ptr + idx * 2 + 1, mask=valid[None, :], other=0)  # nibbles 8..15
    sc = _e4m3_value(tl.load(scales_ptr + idx, mask=valid[None, :], other=0)) * gscale  # [32, G]
    # Produce the 8 k of each word in NATURAL k order so the stores are
    # contiguous 16-byte runs: k offset p (0..7) lives in nibble
    # j = (p & 1) * 4 + (p >> 1)  (inverse of korder 0,2,4,6,1,3,5,7).
    p = tl.arange(0, 8)
    shift = (4 * ((p & 1) * 4 + (p >> 1)))[None, None, :]
    v0 = _e2m1_value((w0[:, :, None] >> shift) & 0xF) * sc[:, :, None]  # [32, G, 8]
    v1 = _e2m1_value((w1[:, :, None] >> shift) & 0xF) * sc[:, :, None]
    base_idx = row[:, None, None] * K + g[None, :, None] * 16 + p[None, None, :]
    m3 = valid[None, :, None]
    tl.store(out_ptr + base_idx, v0.to(tl.float16), mask=m3)
    tl.store(out_ptr + base_idx + 8, v1.to(tl.float16), mask=m3)


def qpn_dequant(
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    gscale: float,
    n: int,
    k: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense fp16 [n, k] weight from the QPN prepack; `out` may be reused."""
    groups = k // 16
    tiles = n // 32
    if out is None:
        out = torch.empty(n, k, dtype=torch.float16, device=qpn_codes.device)
    # Sweep 2026-09-05 (V100, 34816x5120): G=64/8 warps 1.83 ms, G=32/4 2.19 ms.
    groups_per_block = 64
    grid = (tiles, triton.cdiv(groups, groups_per_block))
    codes32 = qpn_codes.view(torch.int32)  # two words per (tile, group, lane)
    _qpn_dequant_kernel[grid](
        codes32, qpn_scales, out, gscale, groups, k,
        GROUPS_PER_BLOCK=groups_per_block, num_warps=8,
    )
    return out
