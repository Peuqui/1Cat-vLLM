# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""check_shared_mem() must judge the worker's own device, not device 0."""

import pytest

from vllm.model_executor.layers.fla.ops import utils


@pytest.fixture
def mixed_rig(monkeypatch):
    # Device 0 is an Ampere-class card, device 1 a V100.
    monkeypatch.setattr(utils, "get_all_max_shared_mem", lambda: [166912, 98304])
    utils.check_shared_mem.cache_clear()
    yield
    utils.check_shared_mem.cache_clear()


def test_uses_current_device(mixed_rig, monkeypatch):
    monkeypatch.setattr(utils.device_torch_lib, "current_device", lambda: 1)
    assert utils.check_shared_mem() is False


def test_first_card_still_reachable_by_index(mixed_rig, monkeypatch):
    monkeypatch.setattr(utils.device_torch_lib, "current_device", lambda: 1)
    assert utils.check_shared_mem(tensor_idx=0) is True
