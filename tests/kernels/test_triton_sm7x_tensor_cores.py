# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The setting that hands Volta and Turing routes to Triton tensor-core kernels."""

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import sm7x

DEVICE = torch.device("cuda", 3)
ON = KernelConfig(triton_sm7x_tensor_cores=True)


def _device(monkeypatch, capability: int) -> None:
    def device_capability(device_id: int | None = None) -> DeviceCapability:
        assert device_id == DEVICE.index
        return DeviceCapability(capability // 10, capability % 10)

    monkeypatch.setattr(
        sm7x.current_platform, "get_device_capability", device_capability
    )


def _compiler(monkeypatch, lowers: bool) -> list[int]:
    probed: list[int] = []

    def probe(capability: int) -> bool:
        probed.append(capability)
        return lowers

    monkeypatch.setattr(sm7x, "lowers_dot_to_mma", probe)
    return probed


def test_off_by_default_without_probing(monkeypatch):
    _device(monkeypatch, 75)
    probed = _compiler(monkeypatch, lowers=False)
    assert not sm7x.triton_sm7x_tensor_cores(KernelConfig(), DEVICE)
    assert probed == []


@pytest.mark.parametrize("capability", [70, 75])
def test_on_with_tensor_core_triton(monkeypatch, capability):
    _device(monkeypatch, capability)
    probed = _compiler(monkeypatch, lowers=True)
    assert sm7x.triton_sm7x_tensor_cores(ON, DEVICE)
    assert probed == [capability]


@pytest.mark.parametrize("capability", [70, 75])
def test_on_with_fma_triton_stops(monkeypatch, capability):
    _device(monkeypatch, capability)
    _compiler(monkeypatch, lowers=False)
    with pytest.raises(RuntimeError, match=f"lowers tl.dot to FMA on sm_{capability}"):
        sm7x.triton_sm7x_tensor_cores(ON, DEVICE)


def test_other_architectures_keep_their_routes(monkeypatch):
    _device(monkeypatch, 80)
    probed = _compiler(monkeypatch, lowers=True)
    assert not sm7x.triton_sm7x_tensor_cores(ON, DEVICE)
    assert probed == []


def test_probe_sees_mma_where_every_triton_emits_it():
    # Ampere's fp16 tl.dot is MMA in any Triton; Volta and Turing depend on the build.
    assert sm7x.lowers_dot_to_mma(80)


def test_setting_changes_the_compilation_hash():
    assert KernelConfig().compute_hash() != ON.compute_hash()
