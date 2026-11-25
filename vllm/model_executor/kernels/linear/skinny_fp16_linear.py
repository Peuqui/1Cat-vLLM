# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny FP16 linear ``y[M, N] = x[M, K] @ W[N, K]^T (+ bias)`` for decode steps.

An unquantized FP16 layer with at most a handful of rows (decode and verify
steps, here 1..8) is bound by reading its weight once. On Volta and Turing
cuBLAS stays far above that: the DeepSeek-V4 router gate (256 x 4096, 2.1 MB)
takes 13.4 us in ``Kernel2`` plus 4.0 us in ``splitKreduce`` on a V100 against
a byte time of about 3 us, and the compressor/indexer products run as
``gemmSN_TN`` with 8 CTAs. Here one program owns ``BLOCK_N`` weight rows and a
slice of K, reads that weight tile once for all rows and runs the dot on tensor
cores. The weight is the tensor the layer already holds ([N, K] row-major, K
contiguous): nothing is packed or copied.

Few output columns cannot fill the card, so K is split over a grid axis
(``split_k``). The partial sums go to a small FP32 workspace and a second launch
adds them in a fixed order, so the result does not depend on scheduling.
Without a split (many columns, or a short K) one launch writes the output.

Accumulation is FP32 over K with a single rounding to the output dtype at the
end, like the cuBLAS kernels it replaces; only the summation order differs, so
results are not bit-identical to cuBLAS. The output is FP16 (the contract of
``F.linear``) or FP32 (the contract of ``torch.mm(..., out_dtype=torch.float32)``
that the DeepSeek-V4 compressors use: the accumulator is stored without the FP16
rounding in between).

Needs the Triton of ``kernel_config.triton_sm7x_tensor_cores`` (stock Triton
lowers ``tl.dot`` to FMA on these cards); ``KernelConfig.use_skinny_fp16_linear``
checks that. The layer asks ``skinny_fp16_linear_layer_decision`` once when its
weights are loaded and keeps the answer; a ``None`` tile means it stays on
cuBLAS. At run time ``torch.ops.vllm.skinny_fp16_linear`` is the one call: it is
opaque to ``torch.compile`` and picks the kernel for 1..8 rows and the old
cuBLAS call for any other batch, because a compiled graph is traced once and
cannot branch on the batch size.
"""

from functools import cache
from typing import NamedTuple

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

# One MMA row tile: more rows than this and cuBLAS is no longer bound by the
# weight, so the batch keeps the cuBLAS call.
SKINNY_FP16_LINEAR_MAX_ROWS = 8
# Largest weight the layer decision admits. Measured on the V100 and the Quadro
# RTX 8000 (benchmark_skinny_fp16_linear.py, 25.11.2026): up to the 16.8 MB of
# the DeepSeek-V4 compressor the best tile beats cuBLAS by 1.0-1.5x (5x on the
# 64-row weights_proj), while the 66 and 67 MB weights of the DSpark markov head
# and GLM's eh_proj stay at or behind cuBLAS (V100 566-584 GB/s against 656-726
# GB/s): a weight that large is streamed by cuBLAS at 70-80 % of the card's
# bandwidth already.
SKINNY_FP16_LINEAR_MAX_WEIGHT_BYTES = 32 * 2**20
SKINNY_FP16_LINEAR_MAX_CHUNKS = 8  # unrolled K chunks per program
SKINNY_FP16_LINEAR_MAX_SPLIT_K = 32
_BLOCK_M = 16  # smallest tl.dot row tile; rows past the batch are masked
_MIN_BLOCK_K = 16  # smallest tl.dot reduction tile; K is a multiple of it
_REDUCE_BLOCK = 1024
_REDUCE_WARPS = 4
_OUTPUT_DTYPES = (torch.float16, torch.float32)


class SkinnyTile(NamedTuple):
    block_n: int  # weight rows (output columns) per program
    block_k: int  # K per dot
    num_warps: int
    split_k: int  # programs along K; the chunk count per program is derived


class SkinnyClass(NamedTuple):
    """Tile shape and grid size of one class of weights."""

    block_n: int
    block_k: int
    num_warps: int
    ctas_per_sm: float  # programs the grid should hold per SM; K is split to get there


class SkinnyArchRule(NamedTuple):
    """Weights up to ``small_n_max`` rows take ``small``, wider ones ``large``."""

    small_n_max: int
    small: SkinnyClass
    large: SkinnyClass


# One rule per compute capability, fitted to the tile sweep of 25.11.2026
# (benchmark_skinny_fp16_linear.py --sweep --rows 5 6 7 on the ten shapes of
# DeepSeek-V4-Flash and GLM-5.3-Flash with N = 64..4096 and K = 1536/4096). The
# worst tile of the rule is 6.5 % (Quadro RTX 8000) and 8.8 % (V100) slower than
# the best sweep tile of its shape, the mean 2.3 % and 4.7 %; no two-class
# rule of this form does better on the sweep data. Re-fit from a new sweep: the
# values are data.
SKINNY_ARCH_RULES: dict[int, SkinnyArchRule] = {
    70: SkinnyArchRule(
        small_n_max=288,
        small=SkinnyClass(block_n=32, block_k=512, num_warps=4, ctas_per_sm=0.5),
        large=SkinnyClass(block_n=32, block_k=128, num_warps=2, ctas_per_sm=1.0),
    ),
    75: SkinnyArchRule(
        small_n_max=256,
        small=SkinnyClass(block_n=16, block_k=256, num_warps=2, ctas_per_sm=0.75),
        large=SkinnyClass(block_n=32, block_k=256, num_warps=4, ctas_per_sm=0.5),
    ),
}


class SkinnyDecision(NamedTuple):
    """The tile when the product is covered, else ``tile`` is None and ``reason``
    says why (for the log line of the layer that takes another route)."""

    tile: SkinnyTile | None
    reason: str


@triton.jit
def _skinny_fp16_linear_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    partial_ptr,
    rows,
    n,
    x_stride_row,
    w_stride_n,
    out_stride_row,
    K_CHUNKS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = cols < n
    row_index = tl.arange(0, BLOCK_M)
    row_ok = row_index < rows
    offs_k = tl.arange(0, BLOCK_K)
    k_start = split * (K_CHUNKS * BLOCK_K)
    x_base = x_ptr + row_index[:, None] * x_stride_row + k_start
    w_base = w_ptr + cols[None, :].to(tl.int64) * w_stride_n + k_start
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    # Unrolled: the loads of every chunk are independent of the accumulator, so
    # they can be in flight together (Volta and Turing have no async copy that
    # a software pipeline could use).
    for chunk in tl.static_range(K_CHUNKS):
        k = chunk * BLOCK_K + offs_k
        w = tl.load(w_base + k[:, None], mask=col_ok[None, :], other=0.0)
        x = tl.load(x_base + k[None, :], mask=row_ok[:, None], other=0.0)
        acc += tl.dot(x, w)
    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(bias_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)[
                None, :
            ]
        tl.store(
            out_ptr + row_index[:, None] * out_stride_row + cols[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=row_ok[:, None] & col_ok[None, :],
        )
    else:
        tl.store(
            partial_ptr + (split * rows + row_index[:, None]) * n + cols[None, :],
            acc,
            mask=row_ok[:, None] & col_ok[None, :],
        )


@triton.jit
def _skinny_fp16_linear_reduce_kernel(
    partial_ptr,
    bias_ptr,
    out_ptr,
    rows,
    n,
    out_stride_row,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = index < rows * n
    row = index // n
    col = index - row * n
    # Fixed order over the splits: the result does not depend on scheduling.
    acc = tl.load(partial_ptr + index, mask=ok, other=0.0)
    for split in tl.static_range(1, SPLIT_K):
        acc += tl.load(partial_ptr + split * rows * n + index, mask=ok, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + col, mask=ok, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + row * out_stride_row + col,
        acc.to(out_ptr.dtype.element_ty),
        mask=ok,
    )


@cache
def skinny_fp16_linear_device_traits(device: torch.device) -> tuple[int, int]:
    """(compute capability as e.g. 70, SM count) of the card the weight lives on;
    the arguments the decisions take."""
    capability = current_platform.get_device_capability(device.index)
    if capability is None:
        raise ValueError(f"no compute capability for {device}")
    return capability.to_int(), num_compute_units(device.index)


def skinny_fp16_linear_tile(
    n: int, k: int, capability: int, sm_count: int
) -> SkinnyDecision:
    """The tile the rule of ``capability`` gives an [n, k] weight on a card with
    ``sm_count`` SMs: the class by N, K halved from its ``block_k`` until it
    divides, then the smallest split of the K chunks (a power of two that divides
    them, leaves a program at most 8 and stays within 32) that brings the grid to
    ``ctas_per_sm`` programs per SM, or the largest split there is."""
    if capability not in SKINNY_ARCH_RULES:
        return SkinnyDecision(None, f"no skinny FP16 tile for capability {capability}")
    rule = SKINNY_ARCH_RULES[capability]
    shape_class = rule.small if n <= rule.small_n_max else rule.large
    block_k = next(
        (
            b
            for b in (512, 256, 128, 64, 32, _MIN_BLOCK_K)
            if b <= shape_class.block_k and k % b == 0
        ),
        None,
    )
    if block_k is None:
        return SkinnyDecision(None, f"K={k} is not a multiple of {_MIN_BLOCK_K}")
    chunks = k // block_k
    splits = [
        split
        for split in (
            2**power for power in range(SKINNY_FP16_LINEAR_MAX_SPLIT_K.bit_length())
        )
        if split <= SKINNY_FP16_LINEAR_MAX_SPLIT_K
        and chunks % split == 0
        and chunks // split <= SKINNY_FP16_LINEAR_MAX_CHUNKS
    ]
    if not splits:
        return SkinnyDecision(
            None,
            f"K={k} cannot be split into at most "
            f"{SKINNY_FP16_LINEAR_MAX_CHUNKS} chunks of {block_k} per program "
            f"within split {SKINNY_FP16_LINEAR_MAX_SPLIT_K}",
        )
    column_tiles = triton.cdiv(n, shape_class.block_n)
    target = shape_class.ctas_per_sm * sm_count
    split_k = next((s for s in splits if column_tiles * s >= target), splits[-1])
    return SkinnyDecision(
        SkinnyTile(shape_class.block_n, block_k, shape_class.num_warps, split_k), ""
    )


def _unsupported_weight(
    weight: torch.Tensor, bias: torch.Tensor | None, out_dtype: torch.dtype
) -> str | None:
    """Why the kernel cannot read this weight, bias and output, or None."""
    if weight.dim() != 2:
        return f"needs weight [N, K], got {tuple(weight.shape)}"
    n, k = weight.shape
    if k % _MIN_BLOCK_K:
        return f"K={k} is not a multiple of {_MIN_BLOCK_K}"
    if weight.dtype != torch.float16:
        return f"needs an FP16 weight, got {weight.dtype}"
    if out_dtype not in _OUTPUT_DTYPES:
        return f"output must be FP16 or FP32, got {out_dtype}"
    if weight.stride(1) != 1:
        return f"K axis must be contiguous: weight {weight.stride()}"
    if weight.stride(0) < k:
        return f"weight rows overlap: row stride {weight.stride(0)} < K={k}"
    if bias is not None:
        if bias.dtype != torch.float16 or bias.shape != (n,) or bias.stride(0) != 1:
            return (
                "bias must be a contiguous FP16 [N] vector, got "
                f"{bias.dtype} {tuple(bias.shape)} stride {bias.stride()}"
            )
        if bias.device != weight.device:
            return f"bias on {bias.device}, weight on {weight.device}"
    return None


def _unsupported_activation(x: torch.Tensor, weight: torch.Tensor) -> str | None:
    """Why the kernel cannot read this activation for ``weight``, or None."""
    if x.dim() != 2:
        return f"needs x [M, K], got {tuple(x.shape)}"
    rows = x.shape[0]
    if not 1 <= rows <= SKINNY_FP16_LINEAR_MAX_ROWS:
        return f"takes 1..{SKINNY_FP16_LINEAR_MAX_ROWS} rows, got {rows}"
    if weight.dim() == 2 and weight.shape[1] != x.shape[1]:
        return f"K mismatch: x {x.shape[1]}, weight {weight.shape[1]}"
    if x.dtype != torch.float16:
        return f"needs FP16 x, got {x.dtype}"
    if x.stride(1) != 1:
        return f"K axis must be contiguous: x {x.stride()}"
    if x.device != weight.device:
        return f"x on {x.device}, weight on {weight.device}"
    return None


def skinny_fp16_linear_layer_decision(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    capability: int,
    sm_count: int,
) -> SkinnyDecision:
    """Whether a layer with this weight takes the skinny kernel and with which
    tile. The one place that decides; the layer calls it once when its weights
    are loaded (``skinny_fp16_linear_device_traits`` gives ``capability`` and
    ``sm_count``) and keeps its cuBLAS route for a ``None`` tile.

    Covered: an FP16 ``weight`` [N, K] with K a multiple of 16, K contiguous
    (any row stride at least K, so a row slice of a merged weight works), at most
    ``SKINNY_FP16_LINEAR_MAX_WEIGHT_BYTES`` of it, an optional FP16 bias [N], FP16
    or FP32 output, a capability with a rule and a K the rule can split.
    """
    reason = _unsupported_weight(weight, bias, out_dtype)
    if reason is not None:
        return SkinnyDecision(None, reason)
    n, k = weight.shape
    if n * k * 2 > SKINNY_FP16_LINEAR_MAX_WEIGHT_BYTES:
        return SkinnyDecision(
            None,
            f"weight of {n * k * 2 / 2**20:.0f} MiB is above "
            f"{SKINNY_FP16_LINEAR_MAX_WEIGHT_BYTES // 2**20} MiB: cuBLAS streams it "
            "at least as fast",
        )
    return skinny_fp16_linear_tile(n, k, capability, sm_count)


def skinny_fp16_linear_decision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    capability: int,
    sm_count: int,
) -> SkinnyDecision:
    """``skinny_fp16_linear_layer_decision`` for a concrete activation ``x``
    [M, K] with 1 <= M <= 8, FP16, K contiguous, on the weight's device."""
    reason = _unsupported_activation(x, weight)
    if reason is not None:
        return SkinnyDecision(None, reason)
    return skinny_fp16_linear_layer_decision(
        weight, bias, out_dtype, capability=capability, sm_count=sm_count
    )


def skinny_fp16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    tile: SkinnyTile,
    bias: torch.Tensor | None = None,
) -> None:
    """``out = x @ weight.T (+ bias)`` for 1..8 rows, into ``out`` [M, N] (FP16 or
    FP32, last axis contiguous, any row stride). ``tile`` comes from a decision
    (or the benchmark's sweep) for this weight; the launcher checks what the
    kernel reads, not the weight-size limit of the decision."""
    reason = _unsupported_activation(x, weight) or _unsupported_weight(
        weight, bias, out.dtype
    )
    if reason is not None:
        raise ValueError(f"skinny FP16 linear: {reason}")
    rows, k = x.shape
    n = weight.shape[0]
    if out.shape != (rows, n) or out.stride(1) != 1 or out.device != x.device:
        raise ValueError(
            f"skinny FP16 linear: out must be a [{rows}, {n}] tensor on {x.device} "
            f"with a contiguous last axis, got {tuple(out.shape)} "
            f"stride {out.stride()} on {out.device}"
        )
    chunk_k = tile.block_k * tile.split_k
    if k % chunk_k:
        raise ValueError(
            f"skinny FP16 linear: K={k} is not a multiple of block_k * split_k = "
            f"{chunk_k}"
        )
    if tile.block_k % _MIN_BLOCK_K or tile.block_n < 16:
        raise ValueError(f"skinny FP16 linear: tile {tile} is below one MMA tile")
    partial = (
        torch.empty((tile.split_k, rows, n), dtype=torch.float32, device=x.device)
        if tile.split_k > 1
        else out
    )
    _skinny_fp16_linear_kernel[(triton.cdiv(n, tile.block_n), tile.split_k)](
        x,
        weight,
        bias if bias is not None else weight,
        out,
        partial,
        rows,
        n,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        K_CHUNKS=k // chunk_k,
        SPLIT_K=tile.split_k,
        HAS_BIAS=bias is not None,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=tile.block_n,
        BLOCK_K=tile.block_k,
        num_warps=tile.num_warps,
        num_stages=1,
    )
    if tile.split_k > 1:
        _skinny_fp16_linear_reduce_kernel[(triton.cdiv(rows * n, _REDUCE_BLOCK),)](
            partial,
            bias if bias is not None else weight,
            out,
            rows,
            n,
            out.stride(0),
            SPLIT_K=tile.split_k,
            HAS_BIAS=bias is not None,
            BLOCK=_REDUCE_BLOCK,
            num_warps=_REDUCE_WARPS,
        )


def _skinny_fp16_linear_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    block_n: int,
    block_k: int,
    num_warps: int,
    split_k: int,
    float32_output: bool,
) -> torch.Tensor:
    """The product of a layer the decision admitted, as a new tensor. 1..8 FP16
    rows take the skinny kernel; any other batch the call the layer made before
    (``F.linear`` for FP16 output, ``torch.mm(..., out_dtype=torch.float32)`` for
    FP32 output). The branch is here, inside the opaque op, because a compiled
    graph holds the branch of the batch it was traced with."""
    rows = x.shape[0]
    if (
        1 <= rows <= SKINNY_FP16_LINEAR_MAX_ROWS
        and x.dtype == torch.float16
        and x.stride(1) == 1
    ):
        out = torch.empty(
            (rows, weight.shape[0]),
            dtype=torch.float32 if float32_output else torch.float16,
            device=x.device,
        )
        skinny_fp16_linear(
            x, weight, out, SkinnyTile(block_n, block_k, num_warps, split_k), bias
        )
        return out
    if float32_output:
        if bias is not None:
            raise ValueError("skinny FP16 linear: FP32 output has no bias")
        return torch.mm(x, weight.T, out_dtype=torch.float32)
    return F.linear(x, weight, bias)


def _skinny_fp16_linear_op_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    block_n: int,
    block_k: int,
    num_warps: int,
    split_k: int,
    float32_output: bool,
) -> torch.Tensor:
    return x.new_empty(
        (x.shape[0], weight.shape[0]),
        dtype=torch.float32 if float32_output else x.dtype,
    )


direct_register_custom_op(
    op_name="skinny_fp16_linear",
    op_func=_skinny_fp16_linear_op,
    fake_impl=_skinny_fp16_linear_op_fake,
)


def skinny_fp16_linear_apply(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    tile: SkinnyTile,
    *,
    float32_output: bool = False,
) -> torch.Tensor:
    """``x @ weight.T (+ bias)`` of a layer whose decision gave ``tile``: ``x`` is
    [M, K], the result a new [M, N] tensor (FP16, or FP32 for ``float32_output``)."""
    return torch.ops.vllm.skinny_fp16_linear(
        x,
        weight,
        bias,
        tile.block_n,
        tile.block_k,
        tile.num_warps,
        tile.split_k,
        float32_output,
    )
