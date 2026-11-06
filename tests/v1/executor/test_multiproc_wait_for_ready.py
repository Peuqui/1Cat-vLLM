# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import multiprocessing.connection
from multiprocessing.connection import Connection
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import (
    UnreadyWorkerProcHandle,
    WorkerProc,
    WorkerProcHandle,
)


def _ready_message(rank: int) -> dict[str, Any]:
    return {
        "status": WorkerProc.READY_STR,
        "handle": f"response-{rank}",
        "peer_response_handles": [],
    }


def _record_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[dict[str, Any], UnreadyWorkerProcHandle]]:
    connected: list[tuple[dict[str, Any], UnreadyWorkerProcHandle]] = []

    def connect(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        connected.append((handles, proc_handle))
        return WorkerProcHandle.from_unready_handle(
            proc_handle, worker_response_mq=None, peer_worker_response_mqs=[]
        )

    monkeypatch.setattr(
        WorkerProc, "wait_for_response_handle_ready", staticmethod(connect)
    )
    return connected


def _handle(rank: int, ready_pipe: Connection) -> UnreadyWorkerProcHandle:
    # The process object is never touched while waiting for READY.
    return UnreadyWorkerProcHandle(
        proc=None,  # type: ignore[arg-type]
        rank=rank,
        ready_pipe=ready_pipe,
    )


def test_wait_for_ready_connects_no_queue_when_a_later_worker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected = _record_connections(monkeypatch)
    ready_reader, ready_writer = multiprocessing.Pipe(duplex=False)
    failing_reader, failing_writer = multiprocessing.Pipe(duplex=False)
    ready_writer.send(_ready_message(0))
    ready_writer.close()

    # Let the failing worker die only after the first worker's READY has been
    # picked up, as in a pipeline boot where the last stage runs out of
    # memory while loading.
    real_wait = multiprocessing.connection.wait
    wait_calls = 0

    def wait_then_fail(connections):
        nonlocal wait_calls
        ready = real_wait(connections)
        wait_calls += 1
        if wait_calls == 1:
            failing_writer.close()
        return ready

    monkeypatch.setattr(multiprocessing.connection, "wait", wait_then_fail)

    with pytest.raises(Exception, match="WorkerProc initialization failed"):
        WorkerProc.wait_for_ready(
            [_handle(0, ready_reader), _handle(1, failing_reader)]
        )

    assert wait_calls == 2
    assert connected == []


def test_wait_for_ready_connects_every_worker_in_rank_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected = _record_connections(monkeypatch)
    handles = []
    for rank in (1, 0):
        reader, writer = multiprocessing.Pipe(duplex=False)
        writer.send(_ready_message(rank))
        writer.close()
        handles.append(_handle(rank, reader))

    workers = WorkerProc.wait_for_ready(handles)

    assert [worker.rank for worker in workers] == [0, 1]
    assert sorted(handle["handle"] for handle, _ in connected) == [
        "response-0",
        "response-1",
    ]
