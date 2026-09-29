# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read safetensors checkpoints with O_DIRECT, past the page cache.

The default loader memory-maps each shard, so every byte it touches lands in
the page cache and stays there. With a checkpoint larger than host RAM that
cache pushes other processes into swap while the model loads. Reading with
O_DIRECT into private buffers leaves the page cache alone, the way llama.cpp's
direct I/O does.

Under pipeline parallelism a memory-mapped shard only reads the pages a stage
touches. A reader that fills buffers itself has to decide up front, so
tensors of decoder layers outside this stage's range are not read at all.
"""

import json
import mmap
import os
import struct
from collections.abc import Callable, Generator

import regex as re
import torch

# O_DIRECT needs offsets, lengths and buffers aligned to the logical block
# size; 4 KiB covers the devices in use.
_ALIGN = 4096
# Tensors closer than this are read in one request with the gap between them.
_MAX_GAP = 1 << 20
# Upper bound for one coalesced read; a larger single tensor gets its own read.
_MAX_RUN = 256 << 20

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# `<prefix>layers.<N>.` where <prefix> is a decoder stack whose N is the global
# layer index. MTP heads (`mtp.layers.N`) and vision towers number their own
# layers and are always read.
_DECODER_LAYER = re.compile(
    r"^(?:|model\.|model\.language_model\.|language_model\.model\.)layers\.(\d+)\."
)


def decoder_layer_filter(start: int, end: int) -> Callable[[str], bool]:
    """Return a predicate that is True for tensors of decoder layers outside
    [start, end), i.e. the tensors another pipeline stage owns."""

    def outside(name: str) -> bool:
        match = _DECODER_LAYER.match(name)
        return match is not None and not start <= int(match.group(1)) < end

    return outside


def _read_header(fd: int) -> tuple[dict, int]:
    head = os.pread(fd, 8, 0)
    (header_len,) = struct.unpack("<Q", head)
    header = json.loads(os.pread(fd, header_len, 8))
    return header, 8 + header_len


def _read_run(fd: int, start: int, end: int) -> tuple[mmap.mmap, int]:
    """Read [start, end) with O_DIRECT; return the buffer and the aligned
    file offset it starts at."""
    begin = start - start % _ALIGN
    stop = -(-end // _ALIGN) * _ALIGN
    # Private, not the default shared mapping: shared anonymous memory is
    # shmem, which only swap can evict and which /proc/self/maps shows as
    # the file /dev/zero.
    buffer = mmap.mmap(-1, stop - begin, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    view = memoryview(buffer)
    done = 0
    while begin + done < end:
        count = os.preadv(fd, [view[done:]], begin + done)
        if count == 0:
            raise EOFError(
                f"safetensors shard ended at offset {begin + done}, "
                f"expected data up to {end}"
            )
        done += count
    view.release()
    return buffer, begin


def direct_io_weights(
    path: str, keep: Callable[[str], bool]
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield the tensors of one safetensors shard for which `keep` is True,
    read with O_DIRECT in coalesced runs in file order."""
    with open(path, "rb", buffering=0) as header_file:
        header, data_start = _read_header(header_file.fileno())

    tensors = []
    for name, info in header.items():
        if name == "__metadata__" or not keep(name):
            continue
        if info["dtype"] not in _DTYPES:
            raise ValueError(
                f"Direct I/O loading does not know safetensors dtype "
                f"{info['dtype']!r} of {name!r} in {path}"
            )
        begin, end = info["data_offsets"]
        tensors.append((data_start + begin, data_start + end, name, info))
    tensors.sort(key=lambda entry: entry[0])

    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        index = 0
        while index < len(tensors):
            run_start, run_end = tensors[index][0], tensors[index][1]
            last = index + 1
            while (
                last < len(tensors)
                and tensors[last][0] - run_end < _MAX_GAP
                and tensors[last][1] - run_start <= _MAX_RUN
            ):
                run_end = max(run_end, tensors[last][1])
                last += 1
            buffer, buffer_start = _read_run(fd, run_start, run_end)
            for begin, end, name, info in tensors[index:last]:
                dtype = _DTYPES[info["dtype"]]
                shape = info["shape"]
                if end == begin:
                    yield name, torch.empty(shape, dtype=dtype)
                    continue
                raw = torch.frombuffer(
                    buffer,
                    dtype=torch.uint8,
                    count=end - begin,
                    offset=begin - buffer_start,
                )
                yield name, raw.view(dtype).reshape(shape)
            index = last
    finally:
        os.close(fd)
