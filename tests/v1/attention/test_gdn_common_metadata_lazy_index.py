# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pure-decode speculative token index is built only when it is read."""

import pytest
import torch

from vllm.v1.attention.ops.gdn_state import compute_common_gdn_attn_metadata


def _common(drafts: list[int], starts: list[int]):
    return compute_common_gdn_attn_metadata(
        num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        num_spec_state_tokens=2,
        legacy_mixed_decode_routing=False,
    )


def test_pure_decode_defers_the_arange(monkeypatch) -> None:
    built: list[int] = []
    original = torch.arange

    def spy(*args, **kwargs):
        built.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", spy)

    common = _common([2, 2], [0, 3, 6])

    assert common is not None
    assert common.spec_token_size == 6
    assert built == []
    values = common.spec_token_indx
    assert built == [6]
    torch.testing.assert_close(
        values, torch.arange(6, dtype=torch.int32), rtol=0, atol=0
    )
    assert common.spec_token_indx is values


@pytest.mark.parametrize("drafts,starts", [([2, -1], [0, 3, 7]), ([-1, 2], [0, 4, 7])])
def test_mixed_batch_keeps_the_stable_sort_index(drafts, starts) -> None:
    common = _common(drafts, starts)

    assert common is not None
    spec_rows = [i for i, d in enumerate(drafts) if d >= 0]
    expected = torch.cat(
        [torch.arange(starts[i], starts[i + 1]) for i in spec_rows]
    ).to(torch.int32)
    assert common.spec_token_size == expected.numel()
    torch.testing.assert_close(common.spec_token_indx.to(torch.int32), expected)


def _item_calls(monkeypatch) -> list[int]:
    calls: list[int] = []
    original = torch.Tensor.item

    def counting(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(torch.Tensor, "item", counting)
    return calls


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "drafts,starts,expected",
    [
        # pure speculative decode
        ([2, 2], [0, 3, 6], (0, 0, 0, 6)),
        # speculative row plus one prefill row
        ([2, -1], [0, 3, 7], (1, 4, 0, 3)),
        # speculative row plus one single-token row
        ([-1, 2], [0, 1, 4], (1, 1, 0, 3)),
    ],
)
def test_token_counts_use_host_numbers_not_tensor_items(
    monkeypatch, legacy, drafts, starts, expected
) -> None:
    calls = _item_calls(monkeypatch)

    common = compute_common_gdn_attn_metadata(
        num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        num_spec_state_tokens=2,
        legacy_mixed_decode_routing=legacy,
    )

    assert common is not None
    assert calls == []
    prefills, prefill_tokens, decodes, spec_tokens = expected
    if legacy and drafts[0] == -1:
        # The single-token non-speculative row is a decode in legacy routing.
        prefills, prefill_tokens, decodes = 0, 0, 1
    assert (
        common.num_prefills,
        common.num_prefill_tokens,
        common.num_decodes,
        common.num_spec_decode_tokens,
    ) == (prefills, prefill_tokens, decodes, spec_tokens)
