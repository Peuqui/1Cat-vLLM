# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hot rows of the PLE disk tier, kept in the offload worker's memory."""

import numpy as np


class PLERowCache:
    """A fixed-size cache for PLE rows read from the mapped checkpoint.

    The table is hash-addressed, so its hot rows -- the frequent n-grams --
    are scattered over all of it, and each shares its 4 KiB page with rows
    nobody reads. Kept here row by row, the budget holds hot rows only, and
    the kernel cannot drop them when the host runs short of memory.

    Set-associative with small use counts: a hit raises its row's count, and a
    row read from disk takes the coldest way of its set only once that count
    has decayed to zero; otherwise it ages the whole set by one. Rows that
    keep coming back stay, while the one-off n-grams of a long prompt pass
    through without displacing them. Not thread-safe: the offload worker's
    request loop is its only user.
    """

    WAYS = 4
    MAX_COUNT = 7

    def __init__(self, budget_bytes: int, row_bytes: int) -> None:
        # Every row carries its int64 id and a one-byte use count.
        slot_bytes = row_bytes + np.dtype(np.int64).itemsize + 1
        num_sets = budget_bytes // (slot_bytes * self.WAYS)
        if num_sets <= 0:
            raise ValueError(
                f"a PLE row cache of {budget_bytes} bytes holds no set of "
                f"{self.WAYS} rows of {row_bytes} bytes"
            )
        self.num_sets = num_sets
        self._ids = np.full((num_sets, self.WAYS), -1, dtype=np.int64)
        self._counts = np.zeros((num_sets, self.WAYS), dtype=np.uint8)
        self._rows = np.empty((num_sets, self.WAYS, row_bytes), dtype=np.uint8)
        self.hits = 0
        self.misses = 0

    @property
    def capacity_rows(self) -> int:
        return self.num_sets * self.WAYS

    def lookup(self, ids: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Copy the cached rows of ``ids`` into ``out``; return the miss mask."""
        sets = ids % self.num_sets
        match = self._ids[sets] == ids[:, None]
        hit = match.any(axis=1)
        hit_sets = sets[hit]
        hit_ways = match[hit].argmax(axis=1)
        out[hit] = self._rows[hit_sets, hit_ways]
        counts = self._counts[hit_sets, hit_ways]
        self._counts[hit_sets, hit_ways] = np.minimum(counts + 1, self.MAX_COUNT)
        num_hits = int(np.count_nonzero(hit))
        self.hits += num_hits
        self.misses += ids.size - num_hits
        return ~hit

    def insert(self, ids: np.ndarray, rows: np.ndarray) -> None:
        """Offer rows read from the checkpoint; a set takes at most one per call."""
        # Duplicates of one row, or two rows competing for one set, would
        # otherwise write the same way twice within a single assignment.
        sets, first = np.unique(ids % self.num_sets, return_index=True)
        ids, rows = ids[first], rows[first]
        counts = self._counts[sets]
        coldest = counts.argmin(axis=1)
        free = counts[np.arange(sets.size), coldest] == 0
        target_sets, target_ways = sets[free], coldest[free]
        self._ids[target_sets, target_ways] = ids[free]
        self._rows[target_sets, target_ways] = rows[free]
        self._counts[target_sets, target_ways] = 1
        # The coldest way of every refused set is above zero, so all its ways
        # are: aging them cannot wrap around.
        self._counts[sets[~free]] -= 1
