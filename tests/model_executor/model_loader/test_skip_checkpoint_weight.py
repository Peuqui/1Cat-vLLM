# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for skipping checkpoint tensors a model does not load, before they
are read from disk."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import compute_local_expert_ids
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)


def _is_target_weight(name: str) -> bool:
    return not name.startswith("mtp.")


@pytest.fixture
def shared_checkpoint(tmp_path):
    """A target checkpoint that also ships a drafter under ``mtp.*``."""
    tensors = {
        "embed.weight": torch.randn(16, 8),
        "layers.0.attn.wq.weight": torch.randn(8, 8),
        "layers.0.ffn.experts.0.w1.weight": torch.randn(8, 8),
        "layers.0.ffn.experts.1.w1.weight": torch.randn(8, 8),
        "mtp.0.main_proj.weight": torch.randn(8, 16),
        "mtp.0.ffn.experts.0.w1.weight": torch.randn(8, 8),
        "mtp.0.ffn.experts.1.w1.weight": torch.randn(8, 8),
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return tmp_path, [str(path)], tensors


@pytest.fixture
def read_names(monkeypatch):
    """Record every tensor name the iterator actually reads from disk."""
    names: list[str] = []
    real_safe_open = weight_utils.safe_open

    class RecordingSafeOpen:
        def __init__(self, *args, **kwargs):
            self._file = real_safe_open(*args, **kwargs)

        def __enter__(self):
            self._handle = self._file.__enter__()
            return self

        def __exit__(self, *exc):
            return self._file.__exit__(*exc)

        def keys(self):
            return self._handle.keys()

        def get_tensor(self, name):
            names.append(name)
            return self._handle.get_tensor(name)

    monkeypatch.setattr(weight_utils, "safe_open", RecordingSafeOpen)
    return names


def test_skipped_weights_are_never_read(shared_checkpoint, read_names):
    _, files, tensors = shared_checkpoint

    loaded = dict(
        safetensors_weights_iterator(files, False, skip_weight=_is_target_weight)
    )

    drafter_names = {name for name in tensors if name.startswith("mtp.")}
    assert set(loaded) == drafter_names
    assert set(read_names) == drafter_names
    for name, tensor in loaded.items():
        assert torch.equal(tensor, tensors[name])


def test_eager_strategy_honours_skip_weight(shared_checkpoint):
    _, files, tensors = shared_checkpoint

    loaded = dict(
        safetensors_weights_iterator(
            files, False, "eager", skip_weight=_is_target_weight
        )
    )

    assert set(loaded) == {name for name in tensors if name.startswith("mtp.")}


def test_skip_weight_combines_with_ep_filter(shared_checkpoint, read_names):
    _, files, _ = shared_checkpoint
    local_expert_ids = compute_local_expert_ids(2, ep_size=2, ep_rank=0)

    loaded = dict(
        safetensors_weights_iterator(
            files,
            False,
            local_expert_ids=local_expert_ids,
            skip_weight=_is_target_weight,
        )
    )

    expected = {"mtp.0.main_proj.weight", "mtp.0.ffn.experts.0.w1.weight"}
    assert set(loaded) == expected
    assert set(read_names) == expected


class _DrafterInSharedCheckpoint(nn.Module):
    def skip_checkpoint_weight(self, name: str) -> bool:
        return _is_target_weight(name)


def test_loader_applies_the_model_skip_rule(shared_checkpoint, read_names):
    folder, _, tensors = shared_checkpoint
    loader = DefaultModelLoader(LoadConfig())
    model_config = SimpleNamespace(model=str(folder), revision=None)

    loaded = dict(loader.get_all_weights(model_config, _DrafterInSharedCheckpoint()))

    drafter_names = {name for name in tensors if name.startswith("mtp.")}
    assert set(loaded) == drafter_names
    assert set(read_names) == drafter_names


def test_loader_reads_everything_without_a_skip_rule(shared_checkpoint, read_names):
    folder, _, tensors = shared_checkpoint
    loader = DefaultModelLoader(LoadConfig())
    model_config = SimpleNamespace(model=str(folder), revision=None)

    loaded = dict(loader.get_all_weights(model_config, nn.Module()))

    assert set(loaded) == set(tensors)
    assert set(read_names) == set(tensors)
