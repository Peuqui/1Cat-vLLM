# Qwen4Exp PLE overflow cascade

## Decision

The Qwen4Exp PLE table is large: 47.684 GiB of FP8 E4M3 rows for
Qwen3.8-Flash-Next, 23.84 GiB per tensor-parallel rank at TP2. Before the
cascade a pre-Ampere deployment could keep the rank's shard in device memory
with a pinned-host remainder, hold the whole table in the PLE offload worker, or
map it from disk in that worker. A host with little RAM loses with each of
them: the split pins every rank's remainder, the worker's table needs the full
47.7 GiB of host memory, and the disk lane reads every row from the checkpoint.

The cascade fills four tiers per rank, fastest first, and places every row
exactly once:

| Tier | Holds | Served by | Budget |
|---|---|---|---|
| 1 device | rows up to the measured device budget | the compute rank, gathered in its graph | measured at load |
| 2 host | pinned host share | the compute rank, through the UVA view | `VLLM_QWEN4EXP_PLE_HOST_GIB` per rank |
| 3 store | the named cards, filled in order; a card may run a pipeline stage | the PLE offload worker, `index_select` on each card | what each card has free once all stages are set up, minus its reserve |
| 4 disk | the mapped checkpoint shards | the PLE offload worker, mmap reads | none, allowed by `VLLM_QWEN4EXP_PLE_DISK=1` |

The table is addressed by hashes, so every row is equally likely to be read and
the split points carry no meaning beyond capacity. Every occupied tier is hit
on every step; the slowest occupied tier sets the cost. A table that fits the
device therefore pins no host memory, and the disk tier is meant for tables
that fit nowhere else.

The feature is off by default. Without its variables the pinned-host split, the
whole-table worker, the disk lane and the hybrid lane are unchanged.

## Configuration

| Variable | Meaning |
|---|---|
| `VLLM_QWEN4EXP_PLE_STORE_DEVICES` | Visible CUDA indices of the store cards, comma-separated, in fill order. Setting it starts the cascade. |
| `VLLM_QWEN4EXP_PLE_STORE_RESERVE_GIB` | Memory each store card keeps free, one value per card. Default 0.5 GiB per card. Raise it for a card that other tenants share. |
| `VLLM_QWEN4EXP_PLE_DISK` | `1` lets the remainder be read from the checkpoint. Starts the cascade on its own, without a store card. |
| `VLLM_QWEN4EXP_PLE_HOST_GIB` | Pinned host share per rank. With the cascade it is used only for rows the device cannot hold. |
| `VLLM_QWEN4EXP_PLE_VRAM_RESERVE_GIB` | Device memory the measured budget keeps free (default 8 % of the card, at most 4 GiB). |
| `VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB` | Host memory the host-share check keeps free (default a quarter of the host). |

The other pipeline stages at PP4, disk as the last tier:

```bash
VLLM_QWEN4EXP_PLE_HOST_GIB=3 \
VLLM_QWEN4EXP_PLE_STORE_DEVICES=1,2,3 \
VLLM_QWEN4EXP_PLE_DISK=1 \
vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 --pipeline-parallel-size 4 ...
```

A spare fifth card at TP2 that also serves a vision model:

```bash
VLLM_QWEN4EXP_PLE_HOST_GIB=2 \
VLLM_QWEN4EXP_PLE_STORE_DEVICES=4 \
VLLM_QWEN4EXP_PLE_STORE_RESERVE_GIB=10.3 \
vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 ...
```

No spare card, little host memory:

```bash
VLLM_QWEN4EXP_PLE_HOST_GIB=2 \
VLLM_QWEN4EXP_PLE_DISK=1 \
vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 ...
```

The reserve has to cover whatever else may still load on a card; the stage
running on a card is accounted for by the worker. The variables are declared in
`vllm/envs.py`, so changing one of them changes the compile-cache key and the
next boot compiles.

Refused before any worker starts: the cascade together with
`VLLM_SM70_QWEN38_HYBRID_PLE` or `VLLM_PLE_DISK_OFFLOAD`, a model without PLE
layers, store reserves without store devices or with a count that does not
match them, store devices with `data_parallel_size` above 1 (the stages report
their claims within one replica only), and an explicit host share that exceeds
available host memory minus the host reserve, divided by the TP size.

## Placement

`plan_ple_placement` in `vllm/models/qwen4_exp/common/ple.py` splits one rank's
rows into `PLEPlacement(vram_rows, host_rows, remote_rows)`:

1. device rows up to the measured budget: utilization budget minus the memory
   already allocated, the KV cache for `max_model_len` and the device reserve,
2. host rows up to the host share,
3. the rest left to the PLE offload worker, or a startup error that names the
   knobs when the cascade is off.

How the worker splits the remote rows between its store cards and the disk is
decided later, once every stage is set up (see Loading).

Without the cascade the host share is taken first and the device holds the
rest, as before. A rank needs at least one resident row, because both resident
gathers always run.

## Per step

Compute rank (`Qwen4ExpNGramEmbedding` and `Qwen4ExpPinnedHostEmbedding`):

- gathers the device and host rows as in the pinned-host split; ids beyond the
  resident rows read row 0 there,
- waits for the worker with `ple_offload_wait` inside the CUDA graph,
- dequantizes the worker's raw FP8 bytes with the same op and scale as the
  resident gathers and merges them by rank-local id with `where`,
- masks foreign vocabulary ranges and runs the tensor-parallel all-reduce.

PLE offload worker (`_remote_lookup`):

- computes the n-gram ids once per data-parallel rank,
- fills one output buffer for all tensor-parallel ranks: the ranks' store and
  disk segments are disjoint in the global id space, so each rank takes only
  its own slots,
- store rows: per card, `index_select` of that card's ids and a blocking copy
  into their slots of the pinned output,
- disk rows: the disk lane's mmap reader (`_gather_mapped_rows`: sorted unique
  ids per shard, `MADV_RANDOM`, thread pool) for the masked slots only.

Greedy outputs are identical to the pre-change path because every row is
dequantized by the same kernel arithmetic, wherever it was stored.

## Loading and registration

- Each rank copies only its device and host rows from the checkpoint
  (`copy_ple_embedding_shard_tiers_`).
- The worker keeps the 128 checkpoint shards file-backed, like the disk lane,
  and holds no anonymous copy of the table.
- Each rank registers a `PLERemotePlacement` (vocabulary range, resident
  rows). The worker refuses missing or inconsistent registrations and binds
  them, but loads nothing yet: a store card may run a pipeline stage that has
  not allocated its KV cache.
- After `capture_model` and before the kernel warm-up (the first step that
  asks the worker for rows), every rank reports what its stage still claims on
  its card: `allocated + peak_activation - reserved`, because the capture
  released the activations. The rank that spawned the worker sends the claims
  over the control pipe. The worker reads each card's free memory, subtracts
  the claim and the reserve, fills the cards in order (a rank's range is split
  at a card boundary), copies the segments from the mapped shards as raw bytes,
  and acknowledges. Rows left over go to disk, or fail the start when the disk
  tier is not allowed.
- Under pipeline parallelism only ranks that own a `PleOffloadLayer` build a
  connector, and the worker drops an inherited `VLLM_PP_LAYER_PARTITION`, which
  would make `get_pp_indices()` refuse its single-stage world.

## Validation

Rig: 2x Quadro RTX 8000 (first pipeline stage, TP2, holds the table), 2x Tesla
V100 (second stage, TP2), 1x Tesla V100 as store card, PCIe Gen3 x4 (the store
card behind a USB4 tunnel), 30 GiB host RAM, checkpoint on an NVMe SSD in a USB
enclosure. `nvidia/Qwen3.8-Flash-Next-NVFP4`, MTP k=4, TP2 x PP2, async
scheduling, `--max-model-len 262144`, `--gpu-memory-utilization 0.95`.

Probe: three prompts plus the first again, greedy, `ignore_eos`, 260 tokens,
SHA-256 of the text against a reference on the pre-change path.

| Configuration | pinned host | store card | disk | `MemAvailable` after start | text | decode vs reference |
|---|---|---|---|---|---|---|
| before: host share 6 GiB | 12 GiB | – | – | 2.7 GiB | reference | – |
| host share 2 GiB, store | 4 GiB | 4.3 GiB | – | 11.8 GiB | 4/4 identical | -2.5 % |
| host share 0, store | 0 | 8.3 GiB | – | 16.0 GiB | 4/4 identical | -2.3 % |
| host share 2 GiB, store 1 GiB, disk | 4 GiB | 1.0 GiB | 3.3 GiB | not recorded | 4/4 identical | -2.3 % |

- Measured device budget: 19.60 and 19.79 GiB of the 23.84 GiB shard.
- Prefill with identical prompts on both sides: unchanged within 1 %.
- Store load at registration: 1.0 GiB in 25.8 s, 4.3 GiB in 84.6 s, 8.3 GiB in
  157.6 s.
- Cold page cache (`drop_caches` before the boot): the worker read the 1.0 GiB
  of store rows as 261,830 single-page major faults and 10 to 56 MiB of disk
  rows per chat request. The store load took 25.8 s (25.1 to 25.8 s in boots
  without dropping the cache), and decode forward steps per second (tok/s
  divided by the MTP acceptance length) were within 0.5 % of a session whose
  cache had not been dropped.

PP4 on the same rig (TP1 x PP4 on the four compute cards, 12 layers each,
2026-09-23), 12 sampled requests of about 29,200 prompt tokens per row after a
discarded first request:

| Host share | Remote rows on | Decode tok/s, mean (range) | Prefill | `MemAvailable` | Swap-out per request |
|---|---|---|---|---|---|
| 12 GiB | disk, 16.8 GiB | 36.0 (32.5 to 38.0) | 13.5 to 14.0 s | 2.7 to 3.3 GiB | every request |
| 12 GiB | stage 1 card, 16.8 GiB | 35.3 (32.8 to 37.6) | 13.5 to 13.6 s | 4.7 to 5.3 GiB | none |
| 3 GiB | spare fifth card, 25.8 GiB | 39.5 (32.7 to 45.8) | 13.5 to 13.6 s | 12 GiB | none |
| 3 GiB | stage 1 + stage 2 cards, 21.5 + 4.3 GiB | 36.9 (32.0 to 43.3) | 13.5 to 13.7 s | 13.7 to 14 GiB | none |

- Greedy outputs of the last row are identical to the disk row; needles 4/4 at
  24,488 and 101,605 tokens.
- The stages grew 148 to 251 MiB beyond their reported claim with the first
  real requests; the default reserve is twice that.

## Limits

- The offload worker holds memory on each first-stage card (184 and 212 MiB on
  the rig above) outside the utilization budget, as described in #530. The
  device reserve usually absorbs it but does not account for it.
- The disk tier is the slowest tier and is read on every step; a larger disk
  share costs more than the measurement above shows.
- The store tier crosses a process and two PCIe hops per step. On the rig above
  the cascade cost 1.7 to 2.5 % of decode against the pre-change path.
- A store card that runs a pipeline stage is shared in time between two
  processes; at PP4 it served decode about 7 % slower than a spare card. Its
  gain is the host memory it frees, not speed.
