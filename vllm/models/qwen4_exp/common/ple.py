# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen4Exp PLE helpers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PLEShardOverlap:
    """Source and destination slices for one checkpoint embedding shard."""

    source_start: int
    destination_start: int
    row_count: int


def compute_ple_shard_overlap(
    *,
    checkpoint_start: int,
    checkpoint_rows: int,
    tp_start: int,
    tp_end: int,
) -> PLEShardOverlap | None:
    """Compute the overlap of a checkpoint shard and one TP vocabulary range."""

    if checkpoint_start < 0 or checkpoint_rows < 0:
        raise ValueError("checkpoint shard bounds must be non-negative")
    if tp_start < 0 or tp_end < tp_start:
        raise ValueError("invalid TP vocabulary range")
    checkpoint_end = checkpoint_start + checkpoint_rows
    overlap_start = max(checkpoint_start, tp_start)
    overlap_end = min(checkpoint_end, tp_end)
    if overlap_start >= overlap_end:
        return None
    return PLEShardOverlap(
        source_start=overlap_start - checkpoint_start,
        destination_start=overlap_start - tp_start,
        row_count=overlap_end - overlap_start,
    )


def copy_ple_embedding_shard_(
    destination: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> int:
    """Copy the overlapping rows of a PLE checkpoint shard into a TP table."""

    if destination.ndim == 0 or loaded_weight.ndim != destination.ndim:
        raise ValueError("destination and loaded weight must have matching ranks")
    if destination.shape[1:] != loaded_weight.shape[1:]:
        raise ValueError(
            "embedding shard dimensions do not match: "
            f"{tuple(destination.shape[1:])} != {tuple(loaded_weight.shape[1:])}"
        )
    if destination.shape[0] < tp_end - tp_start:
        raise ValueError("destination does not cover the requested TP range")
    overlap = compute_ple_shard_overlap(
        checkpoint_start=checkpoint_start,
        checkpoint_rows=loaded_weight.shape[0],
        tp_start=tp_start,
        tp_end=tp_end,
    )
    if overlap is None:
        return 0
    source = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
    target = destination.narrow(0, overlap.destination_start, overlap.row_count)
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))
    return overlap.row_count

@dataclass(frozen=True)
class PLEPlacement:
    """How many PLE table rows live in device memory and how many on the host."""

    vram_rows: int
    host_rows: int

    @property
    def total_rows(self) -> int:
        return self.vram_rows + self.host_rows


def plan_ple_placement(
    *,
    total_rows: int,
    row_bytes: int,
    host_budget_bytes: int,
) -> PLEPlacement:
    """Place as many PLE rows as possible in device memory.

    The table is addressed by hashes, so every row is equally likely to be read
    and the split point carries no meaning beyond capacity: whatever does not
    fit in the device budget goes to the host. Rows are never dropped, so the
    caller must provide a budget large enough for the remainder.
    """

    if total_rows < 0 or row_bytes <= 0:
        raise ValueError("total_rows must be non-negative and row_bytes positive")
    if host_budget_bytes < 0:
        raise ValueError("host budget must be non-negative")
    host_rows = min(total_rows, host_budget_bytes // row_bytes)
    return PLEPlacement(vram_rows=total_rows - host_rows, host_rows=host_rows)


def copy_ple_embedding_shard_split_(
    vram_table: torch.Tensor,
    host_table: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> int:
    """Copy one checkpoint shard into a table split across device and host.

    The split point is a row index in the TP-local range, so both halves are
    plain sub-ranges of the same vocabulary interval and the single-target
    copy above handles each of them unchanged.
    """

    boundary = tp_start + vram_table.shape[0]
    if boundary > tp_end:
        raise ValueError(
            f"device part ({vram_table.shape[0]} rows) exceeds the TP range "
            f"({tp_end - tp_start} rows)"
        )
    copied = 0
    if vram_table.shape[0]:
        copied += copy_ple_embedding_shard_(
            vram_table,
            loaded_weight,
            checkpoint_start=checkpoint_start,
            tp_start=tp_start,
            tp_end=boundary,
        )
    if host_table.shape[0]:
        copied += copy_ple_embedding_shard_(
            host_table,
            loaded_weight,
            checkpoint_start=checkpoint_start,
            tp_start=boundary,
            tp_end=tp_end,
        )
    return copied


def split_ple_embedding_lookup(
    ids: torch.Tensor,
    vram_table: torch.Tensor,
    host_table: torch.Tensor,
) -> torch.Tensor:
    """Gather PLE rows from a table split across device and host memory.

    Both gathers always run: branching on whether any id falls into the host
    part would need a device-to-host sync on every decode step. The unused
    gather reads row 0 and is cheap because the accesses coalesce.

    An empty half is not a special case of the gather but falls back to the
    plain lookup -- ``torch.embedding`` cannot read from a table without rows.
    """

    if host_table.numel() == 0:
        return torch.nn.functional.embedding(ids, vram_table)
    if vram_table.numel() == 0:
        return torch.nn.functional.embedding(ids, host_table)

    boundary = vram_table.shape[0]
    on_host = ids >= boundary
    zero = ids.new_zeros(())
    vram_ids = torch.where(on_host, zero, ids)
    host_ids = torch.where(on_host, ids - boundary, zero)
    # FP8 has no torch.where; the bytes are identical, so select on an int8 view.
    dtype = vram_table.dtype
    gathered_vram = torch.nn.functional.embedding(vram_ids, vram_table.view(torch.int8))
    gathered_host = torch.nn.functional.embedding(host_ids, host_table.view(torch.int8))
    selected = torch.where(on_host.unsqueeze(-1), gathered_host, gathered_vram)
    return selected.view(dtype)


def kv_cache_bytes_for_max_model_len(vllm_config) -> int:
    """Bytes this rank's KV cache needs to serve ``max_model_len``.

    Sums what every KV-owning layer of this pipeline stage declares. The specs
    are the engine's own source of truth for that number -- the same ones the
    allocator consults later -- so the estimate cannot drift from what is
    actually reserved.
    """

    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    total = 0
    for layer in layers.values():
        spec = layer.get_kv_cache_spec(vllm_config)
        if spec is not None:
            total += spec.max_memory_usage_bytes(vllm_config)
    return total


def auto_ple_host_budget_bytes(
    *,
    table_bytes: int,
    device_total_bytes: int,
    device_allocated_bytes: int,
    gpu_memory_utilization: float,
    kv_cache_bytes: int,
    reserve_bytes: int,
) -> int:
    """Host bytes needed so the target context still fits beside the table.

    The cascade keeps the table in device memory and only spills what the
    context claims -- never more. Spilling beyond that would be wasted: with
    pipeline parallelism the weakest stage caps the block count for all of
    them, so surplus room on this rank buys nothing while pinned host memory
    is the scarcest resource on the machine.
    """

    if table_bytes < 0 or device_total_bytes <= 0:
        raise ValueError("table and device sizes must be non-negative")
    usable = int(device_total_bytes * gpu_memory_utilization)
    room_for_table = usable - device_allocated_bytes - kv_cache_bytes - reserve_bytes
    if room_for_table >= table_bytes:
        return 0
    return table_bytes - max(0, room_for_table)


def available_host_bytes() -> int | None:
    """Host memory the kernel currently reports as available, or None."""

    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None
