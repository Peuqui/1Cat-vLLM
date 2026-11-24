# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side contract of the packed per-step index copy in MRV2 prepare_inputs."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import buffer_utils, model_runner
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

DEVICE = torch.device("cpu")
MAX_REQS = 4


def test_packed_copy_is_one_copy_with_aligned_views(monkeypatch) -> None:
    copies: list[int] = []
    original = buffer_utils.async_copy_to_gpu

    def spy(x, out=None, device=None):
        copies.append(len(x))
        return original(x, out=out, device=device)

    monkeypatch.setattr(buffer_utils, "async_copy_to_gpu", spy)
    arrays = [
        np.array([7, 3], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.arange(5, dtype=np.int32),
    ]

    views = buffer_utils.async_copy_to_gpu_packed(arrays, DEVICE)

    assert len(copies) == 1
    for view, array in zip(views, arrays):
        assert view.dtype == torch.int32
        torch.testing.assert_close(view, torch.from_numpy(array), rtol=0, atol=0)
        assert view.data_ptr() % 16 == 0


def _runner(monkeypatch, copies: list[int]) -> SimpleNamespace:
    original = buffer_utils.async_copy_to_gpu

    def spy(x, out=None, device=None):
        copies.append(len(x))
        return original(x, out=out, device=device)

    monkeypatch.setattr(buffer_utils, "async_copy_to_gpu", spy)
    monkeypatch.setattr(model_runner, "async_copy_to_gpu", spy)

    def expand(idx_mapping, total, cu_num_logits, max_expand_len):
        # Reference of the Triton kernel: row r repeats over its logit rows.
        counts = cu_num_logits[1:] - cu_num_logits[:-1]
        expanded = torch.repeat_interleave(idx_mapping, counts)
        local = torch.cat([torch.arange(c, dtype=torch.int32) for c in counts])
        return expanded, local.to(torch.int32)

    monkeypatch.setattr(model_runner, "expand_idx_mapping", expand)
    monkeypatch.setattr(model_runner, "prepare_pos_seq_lens", lambda *a: None)
    monkeypatch.setattr(
        model_runner,
        "combine_sampled_and_draft_tokens",
        lambda *a: torch.zeros(a[-1], dtype=torch.int64),
    )
    cpu_int = lambda: SimpleNamespace(  # noqa: E731
        np=np.zeros(MAX_REQS, dtype=np.int32),
        gpu=torch.zeros(MAX_REQS, dtype=torch.int32),
    )
    return SimpleNamespace(
        device=DEVICE,
        max_num_reqs=MAX_REQS,
        num_speculative_steps=2,
        use_dcp=False,
        prefix_anchor_lens_buffer=None,
        input_buffers=InputBuffers(MAX_REQS, 32, DEVICE),
        req_states=SimpleNamespace(
            req_id_to_index={"a": 2, "b": 0, "c": 3},
            num_computed_tokens_np=np.array([5, 0, 9, 11], dtype=np.int32),
            num_computed_prefill_tokens=np.zeros(MAX_REQS, dtype=np.int32),
            prefill_len=cpu_int(),
            num_computed_tokens=SimpleNamespace(gpu=torch.zeros(MAX_REQS)),
            last_sampled_tokens=None,
            draft_tokens=None,
            all_token_ids=SimpleNamespace(gpu=None),
        ),
    )


@pytest.mark.parametrize(
    "drafts,expected_cu,expected_local",
    [
        ({}, [0, 1, 2, 3], [0, 0, 0]),
        # Rows are sorted by scheduled tokens, then request id insertion order.
        ({"a": [1, 2], "b": [3], "c": []}, [0, 1, 3, 6], [0, 0, 1, 0, 1, 2]),
    ],
)
def test_prepare_inputs_copies_index_arrays_once(
    monkeypatch, drafts, expected_cu, expected_local
) -> None:
    copies: list[int] = []
    runner = _runner(monkeypatch, copies)
    num_scheduled = {"b": 1 + len(drafts.get("b", [])), "c": 1, "a": 3}
    if drafts:
        num_scheduled = {
            "c": 1,
            "b": 2,
            "a": 3,
        }
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        num_scheduled_tokens=num_scheduled,
        scheduled_spec_decode_tokens=drafts,
        has_structured_output_requests=False,
    )
    desc = SimpleNamespace(num_tokens=scheduler_output.total_num_scheduled_tokens)
    desc.num_reqs = 3

    batch = GPUModelRunner.prepare_inputs(runner, scheduler_output, desc)

    order = sorted(num_scheduled, key=num_scheduled.get)
    expected_idx = [runner.req_states.req_id_to_index[r] for r in order]
    assert batch.req_ids == order
    torch.testing.assert_close(
        batch.idx_mapping, torch.tensor(expected_idx, dtype=torch.int32)
    )
    torch.testing.assert_close(
        batch.cu_num_logits, torch.tensor(expected_cu, dtype=torch.int32)
    )
    torch.testing.assert_close(
        batch.expanded_local_pos, torch.tensor(expected_local, dtype=torch.int32)
    )
    assert batch.cu_num_logits.data_ptr() % 16 == 0
    # One copy for all index arrays plus the persistent query_start_loc buffer.
    assert len(copies) == 2
