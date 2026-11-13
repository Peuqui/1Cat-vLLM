# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device capability queries that may run inside a compiled graph."""

import torch

from vllm.platforms import current_platform


@torch.compiler.assume_constant_result
def is_sm70_device(device_id: int | None) -> bool:
    """Whether ``device_id`` (the worker's own device for ``None``) is SM70.

    The capability is static per device, but Dynamo cannot trace the
    platform's cached capability query ("can't handle functions not
    implemented in python"), so a compiled region takes the answer as a
    trace-time constant.
    """
    return current_platform.is_device_capability(70, device_id=device_id)
