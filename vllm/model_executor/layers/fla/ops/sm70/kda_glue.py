# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glue kernels of the GLM-5.3 KDA layer between conv, recurrence and o_proj.

In decode and spec-verify steps the layer launched 15 tiny kernels around the
recurrent kernel: four ``.contiguous()`` copies of q, k, v and beta before it
and the eager ``FusedRMSNormGated.forward_native`` chain (11 kernels) after it.
Both are replaced here by one launch each. The arithmetic is the eager one, op
for op, so the results are bit-identical:

* the packing is a pure copy;
* the norm reproduces torch's fp32 operation order and rounding: ``x * x``
  stored before summing (no FMA contraction), PyTorch's CUDA reduction order
  for a 128-wide row (four sequential values per lane, then an adjacent
  pairwise tree over the 32 lanes), the exact ``mean`` scale 1/128, IEEE
  ``expf`` and division for the sigmoid, and one round-to-nearest cast.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton

_NORM_DIM = 128
_NORM_LANES = 32
_NORM_VECTOR = _NORM_DIM // _NORM_LANES
_PACK_BLOCK = 1024


@triton.jit
def _pair_sum_tree(partial, LANES: tl.constexpr):
    # Adjacent pairwise reduction, as torch's shuffle-down tree combines lanes.
    # Every step adds exactly two values, so its result does not depend on
    # the order the compiler chooses inside the add.
    if LANES == 1:
        return tl.sum(partial, axis=0)
    return _pair_sum_tree(
        tl.sum(tl.reshape(partial, [LANES // 2, 2]), axis=1), LANES // 2
    )


@triton.jit
def _kda_gated_rmsnorm_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    EPS: tl.constexpr,
    DIM: tl.constexpr,
    LANES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = row * DIM
    lane_base = tl.arange(0, LANES) * (DIM // LANES)

    # x.float().pow(2).mean(-1): per lane ((v0 + v1) + v2) + v3 over its four
    # consecutive values, then the pairwise tree over the lanes.
    lane_sum = tl.load(x_ptr + base + lane_base).to(tl.float32)
    lane_sum = lane_sum * lane_sum
    for step in tl.static_range(1, DIM // LANES):
        value = tl.load(x_ptr + base + lane_base + step).to(tl.float32)
        lane_sum = lane_sum + value * value
    total = _pair_sum_tree(lane_sum, LANES)
    rstd = tl.rsqrt(total * (1.0 / DIM) + EPS)

    offsets = tl.arange(0, DIM)
    x = tl.load(x_ptr + base + offsets).to(tl.float32)
    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    gate = tl.load(gate_ptr + base + offsets).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + tldevice.exp(-gate))
    out = ((x * rstd) * weight) * sigmoid
    tl.store(out_ptr + base + offsets, out.to(out_ptr.dtype.element_ty))


def kda_gated_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``FusedRMSNormGated(activation="sigmoid")`` for fp32 ``x``, as one kernel.

    Equals ``(x * rsqrt(x.pow(2).mean(-1) + eps) * weight.float()
    * sigmoid(gate.float())).to(out_dtype)`` evaluated eagerly in that order.
    """
    assert x.dtype == torch.float32 and x.shape[-1] == _NORM_DIM
    assert x.is_contiguous() and gate.is_contiguous() and weight.is_contiguous()
    assert gate.numel() == x.numel() and weight.shape == (_NORM_DIM,)
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    # FMA contraction would fuse x * x into the sum and the final products.
    _kda_gated_rmsnorm_kernel[(x.numel() // _NORM_DIM,)](
        x,
        gate,
        weight,
        out,
        EPS=eps,
        DIM=_NORM_DIM,
        LANES=_NORM_LANES,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _kda_pack_recurrent_inputs_kernel(
    qkv_ptr,
    beta_ptr,
    packed_ptr,
    beta_out_ptr,
    stride_qkv,
    stride_beta,
    num_tokens,
    PROJECTION: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < 3 * PROJECTION
    value = tl.load(qkv_ptr + token * stride_qkv + offsets, mask=mask)
    part = offsets // PROJECTION
    destination = (
        part * (num_tokens * PROJECTION)
        + token * PROJECTION
        + (offsets - part * PROJECTION)
    )
    tl.store(packed_ptr + destination, value, mask=mask)
    if tl.program_id(1) == 0:
        head = tl.arange(0, BLOCK)
        head_mask = head < HEADS
        beta = tl.load(beta_ptr + token * stride_beta + head, mask=head_mask)
        tl.store(beta_out_ptr + token * HEADS + head, beta, mask=head_mask)


def kda_pack_recurrent_inputs(
    qkv: torch.Tensor, beta: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contiguous ``q, k, v`` of shape (1, T, H, D) and ``beta`` (1, T, H).

    One launch instead of the four ``.contiguous()`` copies that
    ``fused_recurrent_kda`` makes of the strided q|k|v split and of beta.
    ``qkv`` is (T, 3 * H * D) and ``beta`` (T, H), both with unit column stride.
    """
    tokens = qkv.size(0)
    projection = num_heads * head_dim
    assert qkv.shape[1] == 3 * projection and qkv.stride(1) == 1
    assert beta.shape == (tokens, num_heads) and beta.stride(1) == 1
    assert num_heads <= _PACK_BLOCK
    packed = torch.empty((3, tokens, projection), dtype=qkv.dtype, device=qkv.device)
    beta_out = torch.empty((tokens, num_heads), dtype=beta.dtype, device=beta.device)
    _kda_pack_recurrent_inputs_kernel[
        (tokens, triton.cdiv(3 * projection, _PACK_BLOCK))
    ](
        qkv,
        beta,
        packed,
        beta_out,
        qkv.stride(0),
        beta.stride(0),
        tokens,
        PROJECTION=projection,
        HEADS=num_heads,
        BLOCK=_PACK_BLOCK,
    )
    q, k, v = (part.view(1, tokens, num_heads, head_dim) for part in packed)
    return q, k, v, beta_out.view(1, tokens, num_heads)
