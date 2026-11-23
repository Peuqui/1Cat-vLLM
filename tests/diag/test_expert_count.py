# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.diag_expert_count import NUM_ROW_BINS, count_distinct_step

NUM_EXPERTS = 16


def _step(topk_ids, real_tokens, max_reqs=4):
    # prepare_inputs fills every entry past the last request with the token count.
    query_start_loc = torch.full((max_reqs + 1,), real_tokens, dtype=torch.int32)
    query_start_loc[0] = 0
    histogram = torch.zeros(NUM_ROW_BINS, NUM_EXPERTS + 1, dtype=torch.int32)
    topk = torch.tensor(topk_ids, dtype=torch.int32)
    count_distinct_step(topk, query_start_loc, histogram)
    return histogram


def test_duplicates_count_once():
    histogram = _step([[1, 2], [2, 3], [1, 3]], real_tokens=3)
    assert histogram.sum() == 1
    assert histogram[2, 3] == 1  # 3 rows -> bin 2, 3 distinct


def test_padded_rows_excluded():
    histogram = _step([[1, 2], [2, 3], [9, 10], [11, 12], [13, 14]], real_tokens=2)
    assert histogram[1, 3] == 1  # only rows 0-1 count: experts 1, 2, 3


def test_no_padding():
    histogram = _step([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], real_tokens=5)
    assert histogram[4, 10] == 1


def test_many_rows_share_last_bin():
    ids = [[row % NUM_EXPERTS, (row + 1) % NUM_EXPERTS] for row in range(12)]
    histogram = _step(ids, real_tokens=12)
    assert histogram[NUM_ROW_BINS - 1, 13] == 1  # experts 0..12


def test_no_valid_rows_adds_nothing():
    assert _step([[1, 2]], real_tokens=0).sum() == 0


def test_accumulates_and_ignores_invalid_ids():
    query_start_loc = torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32)
    histogram = torch.zeros(NUM_ROW_BINS, NUM_EXPERTS + 1, dtype=torch.int32)
    ids = torch.tensor([[1, -1], [1, 5]], dtype=torch.int32)
    count_distinct_step(ids, query_start_loc, histogram)
    count_distinct_step(ids, query_start_loc, histogram)
    assert histogram[1, 2] == 2
