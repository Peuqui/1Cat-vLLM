# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether Triton kernels may count on tensor cores on Volta and Turing.

Stock Triton lowers ``tl.dot`` to FMA before SM80, so SM70-family routes use
cuBLAS or hand-written kernels there. With ``triton_sm7x_tensor_cores`` set,
the routes run Triton kernels instead; every architecture is checked against
the installed compiler first, so the setting cannot select FMA kernels.
"""

import functools
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.config.kernel import KernelConfig


@triton.jit
def _dot_probe_kernel(a_ptr, b_ptr, c_ptr):
    offsets = tl.arange(0, 16)
    tile = offsets[:, None] * 16 + offsets[None, :]
    tl.store(c_ptr + tile, tl.dot(tl.load(a_ptr + tile), tl.load(b_ptr + tile)))


@functools.cache
def lowers_dot_to_mma(capability: int) -> bool:
    """Whether the installed Triton emits MMA for an fp16 ``tl.dot`` on this SM.

    Compiles only; nothing runs on a device.
    """
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = ASTSource(
        _dot_probe_kernel, {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32"}
    )
    ptx = triton.compile(source, target=GPUTarget("cuda", capability, 32)).asm["ptx"]
    return "mma.sync" in ptx


def triton_sm7x_tensor_cores(
    kernel_config: "KernelConfig", device: torch.device
) -> bool:
    """Whether the SM70-family routes on ``device`` take their Triton kernels."""
    if not kernel_config.triton_sm7x_tensor_cores:
        return False
    device_capability = current_platform.get_device_capability(device.index)
    if device_capability is None or device_capability.major != 7:
        return False
    capability = device_capability.to_int()
    if not lowers_dot_to_mma(capability):
        raise RuntimeError(
            "kernel_config.triton_sm7x_tensor_cores is set, but Triton "
            f"{triton.__version__} ({triton.__file__}) lowers tl.dot to FMA on "
            f"sm_{capability}. Install the sm7x Triton or leave the setting off."
        )
    return True
