# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA prefill attention as one gather and two batched matmuls.

The Triton prefill kernel scatters the key reads head by head. Every head of a
token reads the same keys, so gathering them once per token and handing the
rest to the BLAS library does the same work 5-10x faster on V100 and RTX 8000
(64 heads, 64-256 query tokens, 384-640 keys) and lands closer to an fp64
reference than the Triton kernel does.
"""

import math

import torch

WorkspaceSpec = tuple[tuple[int, ...], torch.dtype]


def sparse_prefill_bmm_workspace_specs(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    width: int,
    dtype: torch.dtype,
) -> list[WorkspaceSpec]:
    """Buffers of `sparse_attn_prefill_bmm`, in its argument order."""
    return [
        ((num_tokens, width, head_dim), dtype),
        ((num_tokens, num_heads, width), dtype),
        ((num_tokens, num_heads, width + 1), torch.float32),
        ((num_tokens, num_heads, width + 1), torch.float32),
    ]


def _fit(buffer: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    # Buffers are reserved for the widest index tensor; a narrower one must
    # still see contiguous rows, so cut the flat storage and reshape.
    return buffer.view(-1)[: math.prod(shape)].view(shape)


def sparse_attn_prefill_bmm(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    keys: torch.Tensor,
    scores: torch.Tensor,
    logits: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """q [T, H, D], kv [S, D], indices [T, W] into kv (-1 = unused), lengths
    [T], attn_sink [H]. The four buffers come from the workspace specs and may
    be larger than this call needs. The softmax runs in float32 over the keys
    plus one sink column that only feeds the denominator."""
    num_tokens, width = indices.shape
    num_heads = q.shape[1]
    keys = _fit(keys, (num_tokens, width, kv.shape[-1]))
    scores = _fit(scores, (num_tokens, num_heads, width))
    logits = _fit(logits, (num_tokens, num_heads, width + 1))
    probs = _fit(probs, (num_tokens, num_heads, width + 1))

    unused = indices < 0
    unused |= torch.arange(width, device=indices.device)[None, :] >= lengths[:, None]
    torch.index_select(
        kv, 0, indices.clamp(min=0).view(-1), out=keys.view(-1, kv.shape[-1])
    )
    attend_gathered_keys(
        q, keys, unused, scale, attn_sink, output, scores, logits, probs
    )


def attend_gathered_keys(
    q: torch.Tensor,
    keys: torch.Tensor,
    unused: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    scores: torch.Tensor,
    logits: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """Attention of q [T, H, D] over its own gathered keys [T, W, D]; unused
    [T, W] marks slots that hold no key. Buffers are exactly sized."""
    width = keys.shape[1]
    # An unused slot holds whatever the gather read for it, which need not be
    # a written key: a short prompt has no compressed entries yet, and what
    # the workspace held before may be NaN. Its weight is zero below, but
    # 0 * NaN is NaN.
    keys.masked_fill_(unused[:, :, None], 0)
    torch.baddbmm(scores, q, keys.transpose(1, 2), beta=0, alpha=scale, out=scores)
    logits[..., :width].copy_(scores)
    logits[..., :width].masked_fill_(unused[:, None, :], float("-inf"))
    logits[..., width] = attn_sink
    torch.softmax(logits, dim=-1, out=probs)
    scores.copy_(probs[..., :width])
    torch.bmm(scores, keys, out=output)
