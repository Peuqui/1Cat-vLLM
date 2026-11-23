# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""THROWAWAY diagnostic (never merge): distinct routed experts per MoE layer and step.

Every MoE runner owns a histogram [row bin, distinct experts]. The update is a
registered custom op built from capturable tensor ops only (no host sync), so it
runs inside full CUDA graphs. Rows beyond the real token count (graph padding)
are excluded using ``query_start_loc[-1]``, which prepare_inputs fills with the
real token count for every entry past the last request.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from vllm.utils.torch_utils import direct_register_custom_op

OUTPUT_DIR = Path("/home/mp/.cache/bench-scripts/expert_count_2026-11-23")
FLUSH_EVERY_STEPS = 200
# Row bins: 1..MAX_SEPARATE_ROWS rows get their own bin, more rows share the last one.
MAX_SEPARATE_ROWS = 8
NUM_ROW_BINS = MAX_SEPARATE_ROWS + 1

_HISTOGRAMS: dict[str, torch.Tensor] = {}
_QUERY_START_LOC: list[torch.Tensor] = []
_steps_seen = 0


def count_distinct_step(
    topk_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    histogram: torch.Tensor,
) -> None:
    """Add one step to ``histogram`` [NUM_ROW_BINS, num_experts + 1] in place."""
    num_experts = histogram.shape[1] - 1
    padded_rows = topk_ids.shape[0]
    valid_rows = torch.clamp(query_start_loc[-1:].to(torch.int64), max=padded_rows)
    row_valid = torch.arange(padded_rows, device=topk_ids.device) < valid_rows
    ids = topk_ids.to(torch.int64)
    in_range = (ids >= 0) & (ids < num_experts) & row_valid[:, None]
    # Everything that must not count lands in the dump slot ``num_experts``.
    slots = torch.where(in_range, ids, torch.full_like(ids, num_experts))
    presence = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    flat_slots = slots.reshape(-1)
    presence.scatter_(0, flat_slots, torch.ones_like(flat_slots, dtype=torch.int32))
    distinct = presence[:num_experts].sum(dtype=torch.int64)
    row_bin = torch.clamp(valid_rows - 1, 0, MAX_SEPARATE_ROWS)[0]
    cell = row_bin * histogram.shape[1] + distinct
    # A step without valid rows adds nothing.
    histogram.view(-1).index_add_(
        0, cell.reshape(1), (valid_rows > 0).to(histogram.dtype)
    )


def _count_impl(
    topk_ids: torch.Tensor, query_start_loc: torch.Tensor, histogram: torch.Tensor
) -> None:
    count_distinct_step(topk_ids, query_start_loc, histogram)


def _count_fake(
    topk_ids: torch.Tensor, query_start_loc: torch.Tensor, histogram: torch.Tensor
) -> None:
    return None


direct_register_custom_op(
    op_name="diag_expert_count",
    op_func=_count_impl,
    mutates_args=["histogram"],
    fake_impl=_count_fake,
)


def register_layer(layer_name: str, num_experts: int) -> torch.Tensor:
    """Allocate the histogram eagerly (buffers created inside graph capture would
    live in the graph pool and be recycled afterwards); the runner keeps it."""
    key = layer_name
    duplicate = 1
    while key in _HISTOGRAMS:
        duplicate += 1
        key = f"{layer_name}#{duplicate}"
    histogram = torch.zeros(
        NUM_ROW_BINS,
        num_experts + 1,
        dtype=torch.int32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    _HISTOGRAMS[key] = histogram
    return histogram


def set_query_start_loc(query_start_loc: torch.Tensor) -> None:
    _QUERY_START_LOC.clear()
    _QUERY_START_LOC.append(query_start_loc)


def record(histogram: torch.Tensor, topk_ids: torch.Tensor) -> None:
    torch.ops.vllm.diag_expert_count(topk_ids, _QUERY_START_LOC[0], histogram)


def reset() -> None:
    for histogram in _HISTOGRAMS.values():
        histogram.zero_()


def write(pp_rank: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "row_bins": f"index i < {MAX_SEPARATE_ROWS}: i+1 rows, last: more rows",
        "layers": {
            key: histogram.cpu().tolist() for key, histogram in _HISTOGRAMS.items()
        },
    }
    temporary = OUTPUT_DIR / f"rank{pp_rank}.json.tmp"
    temporary.write_text(json.dumps(payload))
    temporary.replace(OUTPUT_DIR / f"rank{pp_rank}.json")


def on_step(pp_rank: int) -> None:
    global _steps_seen
    _steps_seen += 1
    if _steps_seen % FLUSH_EVERY_STEPS == 0:
        write(pp_rank)
