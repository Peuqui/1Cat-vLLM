# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

import mmap
import os
import socket
import weakref
from collections.abc import Generator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.platforms import current_platform
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

# cudaHostRegisterPortable | cudaHostRegisterMapped
_HOST_REGISTER_FLAGS = 3


class _HostBuffer:
    """Exactly sized pinned host memory backed by a memfd.

    ``Tensor.pin_memory()`` goes through the caching host allocator, which
    rounds every request up to a power of two and pins all of it: a 1.18 GiB
    embedding table held 2 GiB of locked host memory. Processes on one host
    that offload the same tensor (an embedding table a drafter replicates on
    the last pipeline stage) map one memfd instead of holding a copy each.
    The memory goes away with the last process that maps it.
    """

    def __init__(self, nbytes: int, share_key: str | None):
        self.fd = os.memfd_create("vllm-uva-offload", os.MFD_CLOEXEC)
        os.ftruncate(self.fd, nbytes)
        self.created = True
        self.owner_pid = os.getpid()
        if share_key is not None:
            mine = f"{os.getpid()}:{self.fd}"
            store = dist.distributed_c10d._get_default_store()
            owner = store.compare_set(share_key, "", mine).decode()
            if owner != mine:
                # Another process created it first; its fd stays open for the
                # lifetime of that process.
                os.close(self.fd)
                pid, fd = owner.split(":")
                self.fd = os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR | os.O_CLOEXEC)
                self.created = False
                self.owner_pid = int(pid)
        self.mapping = mmap.mmap(self.fd, nbytes)
        self.tensor = torch.frombuffer(self.mapping, dtype=torch.uint8, count=nbytes)
        pointer = self.tensor.data_ptr()
        error = torch.cuda.cudart().cudaHostRegister(
            pointer, nbytes, _HOST_REGISTER_FLAGS
        )
        if error != torch.cuda.cudart().cudaError.success:
            raise RuntimeError(f"cudaHostRegister of {nbytes} bytes failed: {error}")
        # Unregister if the buffer is dropped while the process lives on; at
        # exit the CUDA context goes first and takes the registration along.
        finalizer = weakref.finalize(
            self, torch.cuda.cudart().cudaHostUnregister, pointer
        )
        finalizer.atexit = False


def _share_key(module: nn.Module, qualified_name: str, p: torch.Tensor) -> str | None:
    """Key under which processes on this host share an offloaded tensor, or
    None. Only embedding tables are shared: pipeline stages that hold the
    same table (the first stage and a drafter's replica on the last one) load
    it from the same checkpoint tensor and never transform it."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.distributed.parallel_state import get_dp_group
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    if not isinstance(module, VocabParallelEmbedding) or not dist.is_initialized():
        return None
    return (
        f"vllm.uva_offload/{socket.gethostname()}/dp{get_dp_group().rank_in_group}"
        f"/tp{get_tensor_model_parallel_rank()}/{qualified_name}"
        f"/{tuple(p.shape)}/{p.dtype}"
    )


class UVAOffloader(BaseOffloader):
    """Offloader using Unified Virtual Addressing (UVA) for zero-copy access.

    This offloader moves parameters to pinned CPU memory and creates CUDA views
    using UVA. The GPU can then directly access the CPU memory without explicit
    transfers, at the cost of PCIe bandwidth (slower than GPU memory).

    When UVA is disabled via env var, falls back to a functional_call-based
    approach that moves parameters on-demand.

    Args:
        cpu_offload_max_bytes: Maximum bytes to offload to CPU.
        cpu_offload_params: Set of parameter name segments to selectively
            offload. If empty, all parameters are eligible up to the byte limit.
    """

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
    ):
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0
        self.cpu_offload_params = cpu_offload_params or set()

        self.pin_memory = should_pin_memory()
        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )
        self._host_buffers: list[_HostBuffer] = []

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading."""
        modules = [self._maybe_offload_to_cpu(module) for module in modules_generator]
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return modules

    def wrap_module(
        self, module: nn.Module, *, parameter_prefix: str = ""
    ) -> nn.Module:
        """Apply UVA offloading to one component with its qualified prefix."""
        previous_offload_bytes = self.cpu_offload_bytes
        module = self._maybe_offload_to_cpu(module, parameter_prefix=parameter_prefix)
        if self.cpu_offload_bytes > previous_offload_bytes:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return module

    def _is_parameter_selected(self, name: str, parameter_prefix: str = "") -> bool:
        if not self.cpu_offload_params:
            return True
        qualified_name = f"{parameter_prefix}.{name}" if parameter_prefix else name
        return any(
            f".{param}." in f".{qualified_name}." for param in self.cpu_offload_params
        )

    def _pinned_host_copy(
        self,
        module: nn.Module,
        name: str,
        data: torch.Tensor,
        *,
        parameter_prefix: str,
    ) -> torch.Tensor:
        """Pinned host tensor for one parameter, exactly as large as it is.

        The process that creates the buffer copies the parameter's current
        values into it; a process attaching to a buffer another one created
        leaves its contents alone, since that process may already be loading
        the checkpoint into it."""
        qualified_name = f"{parameter_prefix}.{name}" if parameter_prefix else name
        owner_name, _, _ = name.rpartition(".")
        owner = module.get_submodule(owner_name) if owner_name else module
        nbytes = data.numel() * data.element_size()
        buffer = _HostBuffer(nbytes, _share_key(owner, qualified_name, data))
        self._host_buffers.append(buffer)
        host = buffer.tensor.view(data.dtype).view(data.shape)
        if buffer.created:
            host.copy_(data)
        else:
            logger.info(
                "UVA offload: %s (%s) maps the host copy of process %d.",
                qualified_name,
                format_gib(nbytes),
                buffer.owner_pid,
            )
        return host

    def _maybe_offload_to_cpu(
        self, module: nn.Module, *, parameter_prefix: str = ""
    ) -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            # Add dots to require exact path-segment matches. A qualified
            # prefix lets selectors such as ``visual`` target an entire tower
            # even though ``module.named_parameters()`` returns relative names.
            if not self._is_parameter_selected(name, parameter_prefix):
                continue

            # A component may contain layers already visited by make_layers().
            # Do not copy or account for the same UVA parameter twice when the
            # enclosing component is registered afterwards.
            if getattr(p, "_vllm_is_uva_offloaded", False):
                continue

            if self.pin_memory and current_platform.is_cuda_alike():
                cpu_data = self._pinned_host_copy(
                    module, name, p.data, parameter_prefix=parameter_prefix
                )
            else:
                cpu_data = p.data.to(device="cpu")
                if self.pin_memory:
                    cpu_data = cpu_data.pin_memory()

            if not self.uva_offloading:
                p.data = cpu_data
            else:
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True

            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters and not self.uva_offloading:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device,
                    # it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }

                # set `tie_weights=False` as tied weights in original model
                # become untied when calling .to(device) individually
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    tie_weights=False,
                )
                module.forward = forward
                return output

            module.forward = forward

        return module
