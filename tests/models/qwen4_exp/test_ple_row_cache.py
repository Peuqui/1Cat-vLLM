# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

from vllm.models.qwen4_exp.common.ple_row_cache import PLERowCache

ROW_BYTES = 160
SLOT_BYTES = ROW_BYTES + 9


def _table(rows: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, 256, (rows, ROW_BYTES), dtype=np.uint8
    )


def _gather(cache: PLERowCache, table: np.ndarray, ids: np.ndarray):
    """Serve ids as the disk tier does: cache first, the table for the rest."""
    out = np.empty((ids.size, ROW_BYTES), dtype=np.uint8)
    miss = cache.lookup(ids, out)
    out[miss] = table[ids[miss]]
    cache.insert(ids[miss], table[ids[miss]])
    return out, miss


def test_row_cache_serves_exact_rows() -> None:
    table = _table(20_000)
    cache = PLERowCache(64 * PLERowCache.WAYS * SLOT_BYTES, ROW_BYTES)
    rng = np.random.default_rng(1)
    for _ in range(100):
        ids = rng.integers(0, 2_000, 300)
        out, _ = _gather(cache, table, ids)
        assert np.array_equal(out, table[ids])
    assert cache.hits > 0
    assert cache.hits + cache.misses == 100 * 300


def test_row_cache_hits_rows_it_has_seen() -> None:
    table = _table(1_000)
    cache = PLERowCache(1 << 20, ROW_BYTES)
    ids = np.arange(200)
    assert _gather(cache, table, ids)[1].all()
    assert not _gather(cache, table, ids)[1].any()


def test_row_cache_takes_one_row_per_set_per_call() -> None:
    # Duplicates and rows of one set within a batch must not write a way twice.
    table = _table(1_000)
    cache = PLERowCache(8 * PLERowCache.WAYS * SLOT_BYTES, ROW_BYTES)
    ids = np.array([3, 3, 3 + cache.num_sets, 5, 5])
    _gather(cache, table, ids)
    out, miss = _gather(cache, table, ids)
    assert np.array_equal(out, table[ids])
    # Set 3 took one of its two candidates, set 5 its only row.
    assert miss.tolist() == [False, False, True, False, False]


def test_recurring_rows_stay_while_one_off_rows_pass_through() -> None:
    # Two hot rows per set leave two ways for the traffic of a long prompt.
    cache = PLERowCache(256 * PLERowCache.WAYS * SLOT_BYTES, ROW_BYTES)
    table = _table(40 * cache.capacity_rows)
    sets = np.arange(cache.num_sets)
    hot = np.concatenate((sets, sets + cache.num_sets))
    for _ in range(PLERowCache.MAX_COUNT):
        _gather(cache, table, hot)

    one_off = np.arange(4 * cache.num_sets, 36 * cache.num_sets)
    for batch in np.array_split(one_off, 64):
        _gather(cache, table, batch)
        out, miss = _gather(cache, table, hot)
        assert not miss.any()
        assert np.array_equal(out, table[hot])


def test_rows_nobody_reads_again_make_room() -> None:
    cache = PLERowCache(16 * PLERowCache.WAYS * SLOT_BYTES, ROW_BYTES)
    table = _table(64 * cache.capacity_rows)
    old = np.arange(cache.capacity_rows)
    _gather(cache, table, old)
    # Each refused offer ages a set by one; a once-read row is evictable after
    # one, so a second wave of new rows takes the cache over.
    new = np.arange(cache.capacity_rows, 2 * cache.capacity_rows)
    for _ in range(PLERowCache.WAYS * 2):
        _gather(cache, table, new)
    assert not _gather(cache, table, new)[1].any()


@pytest.mark.parametrize("budget", [0, PLERowCache.WAYS * SLOT_BYTES - 1])
def test_row_cache_refuses_a_budget_without_one_set(budget: int) -> None:
    with pytest.raises(ValueError, match="holds no set"):
        PLERowCache(budget, ROW_BYTES)
