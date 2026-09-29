# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from vllm.model_executor.model_loader import direct_io
from vllm.model_executor.model_loader.default_loader import (
    _pipeline_stage_layer_range,
)
from vllm.model_executor.model_loader.direct_io import (
    decoder_layer_filter,
    direct_io_weights,
)
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)


def _checkpoint(path) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    tensors = {
        "model.language_model.layers.0.w": torch.randn(33, 17),
        "model.language_model.layers.1.w": torch.randn(64, 64).half(),
        "model.language_model.layers.2.w": torch.randn(5, 7).bfloat16(),
        "layers.3.ffn.w": torch.randn(9, 11).to(torch.float8_e4m3fn),
        "layers.3.ffn.scale": torch.randn(3, 4).to(torch.float8_e8m0fnu),
        "mtp.layers.0.w": torch.randint(0, 255, (129,), dtype=torch.uint8),
        "model.visual.blocks.0.w": torch.randint(-5, 5, (7, 3), dtype=torch.int8),
        "model.language_model.embed_tokens.weight": torch.arange(12).reshape(3, 4),
        "empty": torch.empty(0, 4),
    }
    save_file(tensors, str(path))
    return tensors


def _reference(path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt") as f:
        return {name: f.get_tensor(name) for name in f.keys()}  # noqa: SIM118


def _assert_same(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]):
    assert got.keys() == want.keys()
    for name, tensor in want.items():
        assert got[name].dtype == tensor.dtype, name
        assert got[name].shape == tensor.shape, name
        assert torch.equal(got[name].view(torch.uint8), tensor.view(torch.uint8))


@pytest.mark.parametrize("small_runs", [False, True], ids=["one_run", "many_runs"])
def test_direct_io_matches_safe_open(tmp_path, monkeypatch, small_runs):
    path = tmp_path / "shard.safetensors"
    _checkpoint(path)
    if small_runs:
        # Force a separate O_DIRECT request for almost every tensor.
        monkeypatch.setattr(direct_io, "_MAX_RUN", 1)
        monkeypatch.setattr(direct_io, "_MAX_GAP", 0)

    got = dict(direct_io_weights(str(path), lambda name: True))

    _assert_same(got, _reference(path))


def test_direct_io_reads_only_kept_tensors(tmp_path):
    path = tmp_path / "shard.safetensors"
    _checkpoint(path)
    other_stage = decoder_layer_filter(1, 3)

    got = dict(direct_io_weights(str(path), lambda name: not other_stage(name)))

    owned_by_other_stages = {
        "model.language_model.layers.0.w",
        "layers.3.ffn.w",
        "layers.3.ffn.scale",
    }
    want = {
        name: tensor
        for name, tensor in _reference(path).items()
        if name not in owned_by_other_stages
    }
    _assert_same(got, want)


def test_decoder_layer_filter_leaves_other_stacks_alone():
    outside = decoder_layer_filter(4, 8)
    assert outside("layers.2.attn.w")
    assert outside("model.layers.9.mlp.w")
    assert outside("model.language_model.layers.8.w")
    assert not outside("model.language_model.layers.4.w")
    assert not outside("layers.7.ffn.w")
    # MTP heads and vision towers number their own layers.
    assert not outside("mtp.layers.0.w")
    assert not outside("mtp.0.layers.1.w")
    assert not outside("model.visual.layers.0.w")
    assert not outside("model.language_model.embed_tokens.weight")


class _Stack(nn.Module):
    def __init__(self, start: int, end: int, total: int):
        super().__init__()
        self.start_layer, self.end_layer = start, end
        self.layers = nn.ModuleList(nn.Identity() for _ in range(total))


@pytest.mark.parametrize(
    "start,end,expected", [(0, 10, None), (3, 7, (3, 7)), (0, 4, (0, 4))]
)
def test_pipeline_stage_layer_range(start, end, expected):
    model = nn.Sequential(_Stack(start, end, 10))
    assert _pipeline_stage_layer_range(model) == expected


def test_pipeline_stage_layer_range_without_decoder_stack():
    assert _pipeline_stage_layer_range(nn.Linear(2, 2)) is None


def _file_backed(tensor: torch.Tensor) -> bool:
    address = tensor.data_ptr()
    with open("/proc/self/maps") as mappings:
        for line in mappings:
            fields = line.split(maxsplit=5)
            start, end = (int(value, 16) for value in fields[0].split("-"))
            if start <= address < end:
                return (
                    len(fields) == 6
                    and fields[5].startswith("/")
                    and not (fields[5].startswith("/dev/zero"))
                )
    return False


def test_mapped_weights_stay_file_backed_under_direct_io(tmp_path):
    path = tmp_path / "shard.safetensors"
    _checkpoint(path)
    mapped = {"model.language_model.layers.1.w", "mtp.layers.0.w"}

    got = dict(
        safetensors_weights_iterator(
            [str(path)],
            use_tqdm_on_load=False,
            safetensors_load_strategy="direct",
            map_weight=mapped.__contains__,
        )
    )

    _assert_same(got, _reference(path))
    for name, tensor in got.items():
        if tensor.numel():
            assert _file_backed(tensor) == (name in mapped), name
