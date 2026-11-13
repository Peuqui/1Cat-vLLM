# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.compiled_capability import is_sm70_device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sm70_query_compiles_where_the_platform_query_does_not():
    x = torch.zeros(4, device="cuda")

    @torch.compile(fullgraph=True, backend="eager")
    def platform_query(t):
        return t + 1 if current_platform.is_device_capability(70) else t - 1

    @torch.compile(fullgraph=True, backend="eager")
    def compiled_query(t):
        return t + 1 if is_sm70_device(t.device.index) else t - 1

    # The cached platform query cannot be traced; this is why the helper exists.
    with pytest.raises(torch._dynamo.exc.Unsupported):
        platform_query(x)
    expected = 1.0 if current_platform.is_device_capability(70) else -1.0
    assert torch.equal(compiled_query(x), torch.full_like(x, expected))
