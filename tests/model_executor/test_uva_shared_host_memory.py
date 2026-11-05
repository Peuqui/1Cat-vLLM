# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactly sized, shareable pinned host buffers of the UVA offloader."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.model_executor.offloader.uva import _HostBuffer
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cudaHostRegister needs CUDA"
)

# Not a power of two, so a rounding allocator would show.
NBYTES = 3 * (1 << 20) + 4096 * 7


def _mapping_bytes(pointer: int) -> int:
    with open("/proc/self/maps") as maps:
        for line in maps:
            start, end = (int(x, 16) for x in line.split()[0].split("-"))
            if start <= pointer < end:
                return end - start
    raise AssertionError("pointer not mapped")


def test_host_buffer_is_exact_and_pinned():
    torch.zeros(1, device="cuda")
    buffer = _HostBuffer(NBYTES, share_key=None)
    assert buffer.created
    assert buffer.tensor.numel() == NBYTES
    assert buffer.tensor.is_pinned()
    assert _mapping_bytes(buffer.tensor.data_ptr()) == NBYTES

    host = buffer.tensor.view(torch.float16)
    host.copy_(torch.arange(host.numel(), dtype=torch.float32).to(torch.float16))
    device = get_accelerator_view_from_cpu_tensor(host)
    assert device.is_cuda
    assert torch.equal(device.cpu(), host)


def _share(rank: int, port: int, results) -> None:
    torch.cuda.set_device(0)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank
    )
    buffer = _HostBuffer(NBYTES, share_key="test/shared-table")
    if buffer.created:
        buffer.tensor.fill_(7)
    dist.barrier()
    results[rank] = (
        buffer.created,
        buffer.owner_pid,
        os.getpid(),
        int(buffer.tensor.sum()),
    )
    dist.barrier()
    dist.destroy_process_group()


def test_host_buffer_is_shared_between_processes():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    manager = mp.get_context("spawn").Manager()
    results = manager.dict()
    mp.start_processes(_share, args=(port, results), nprocs=2, start_method="spawn")
    (created_a, owner_a, pid_a, sum_a), (created_b, owner_b, pid_b, sum_b) = (
        results[0],
        results[1],
    )
    # Exactly one process created the memfd; both map it.
    assert created_a != created_b
    creator_pid = pid_a if created_a else pid_b
    assert owner_a == owner_b == creator_pid
    assert sum_a == sum_b == 7 * NBYTES
