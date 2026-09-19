# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The split-K QK-D decode branch of the generic DeepSeek V4 sparse MLA impl
must return what the ragged decode kernel returns for the same metadata."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.models.deepseek_v4.amd import rocm
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or torch.cuda.get_device_capability() not in rocm._QK_DSPLIT_DECODE_CAPABILITIES,
    reason="split-K QK-D decode is enabled on sm70/sm75 CUDA devices only",
)

HEAD_DIM = 512
NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
NUM_TOKENS = 6
WINDOW = 128
CACHE_BLOCK_SIZE = 256


def _make_cache(num_rows: int, block_size: int) -> torch.Tensor:
    num_blocks = -(-num_rows // block_size)
    cache = torch.zeros(
        (num_blocks, block_size * 584), dtype=torch.uint8, device="cuda"
    )
    data = cache[:, : block_size * 576].view(num_blocks, block_size, 576)
    nope = torch.randn((num_blocks, block_size, NOPE_HEAD_DIM), device="cuda")
    data[:, :, :NOPE_HEAD_DIM].copy_(nope.to(torch.float8_e4m3fn).view(torch.uint8))
    rope = 0.125 * torch.randn(
        (num_blocks, block_size, ROPE_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    data[:, :, NOPE_HEAD_DIM:].copy_(
        rope.view(torch.uint8).reshape_as(data[:, :, NOPE_HEAD_DIM:])
    )
    cache[:, block_size * 576 :].fill_(124)
    return cache.view(num_blocks, block_size, 584)


def _swa_metadata() -> MagicMock:
    metadata = MagicMock()
    metadata.num_decodes = 1
    metadata.num_decode_tokens = NUM_TOKENS
    indices = torch.full((NUM_TOKENS, 1, WINDOW), -1, dtype=torch.int32, device="cuda")
    lens = torch.tensor([WINDOW, 1, 17, 64, 100, WINDOW], dtype=torch.int32).cuda()
    for row, length in enumerate(lens.tolist()):
        indices[row, 0, :length] = torch.randperm(WINDOW, device="cuda")[:length]
    metadata.decode_swa_indices = indices
    metadata.decode_swa_lens = lens
    ragged, indptr = build_ragged_indices_from_dense(indices, lens)
    metadata.decode_swa_ragged_indices = ragged
    metadata.decode_swa_ragged_indptr = indptr
    metadata.token_to_req_indices = torch.zeros(
        NUM_TOKENS, dtype=torch.int32, device="cuda"
    )
    is_valid = torch.ones(NUM_TOKENS, dtype=torch.bool, device="cuda")
    is_valid[-1] = False
    metadata.is_valid_token = is_valid
    return metadata


def _c4_case(layer: MagicMock) -> tuple[MagicMock, torch.Tensor]:
    topk = 512
    compressed_block = CACHE_BLOCK_SIZE // 4
    num_blocks = 12
    local = torch.full((NUM_TOKENS, topk), -1, dtype=torch.int32, device="cuda")
    for row, length in enumerate([topk, 3, 200, topk, 511, topk]):
        local[row, :length] = torch.randperm(
            num_blocks * compressed_block, device="cuda"
        )[:length]
    layer.compress_ratio = 4
    layer.topk_indices_buffer = local
    metadata = MagicMock()
    metadata.block_size = CACHE_BLOCK_SIZE
    metadata.block_table = torch.randperm(num_blocks, device="cuda").to(torch.int32)[
        None, :
    ]
    return metadata, _make_cache(num_blocks * compressed_block, compressed_block)


def _c128_case(layer: MagicMock) -> tuple[MagicMock, torch.Tensor]:
    width = 256
    compressed_block = CACHE_BLOCK_SIZE // 128
    layer.compress_ratio = 128
    layer.topk_indices_buffer = None
    indices = torch.full((NUM_TOKENS, 1, width), -1, dtype=torch.int32, device="cuda")
    lens = torch.tensor([141, 141, 1, 140, 141, 0], dtype=torch.int32).cuda()
    for row, length in enumerate(lens.tolist()):
        indices[row, 0, :length] = torch.randperm(width, device="cuda")[:length]
    metadata = MagicMock()
    metadata.block_size = CACHE_BLOCK_SIZE
    metadata.c128a_global_decode_topk_indices = indices
    metadata.c128a_decode_topk_lens = lens
    ragged, indptr = build_ragged_indices_from_dense(indices, lens)
    metadata.c128a_decode_topk_ragged_indices = ragged
    metadata.c128a_decode_topk_ragged_indptr = indptr
    return metadata, _make_cache(width, compressed_block)


@pytest.mark.parametrize("case", ["swa_only", "c4", "c128"])
@pytest.mark.parametrize("num_heads", [8, 64])
@torch.inference_mode()
def test_qk_dsplit_decode_matches_ragged_decode(case: str, num_heads: int) -> None:
    torch.manual_seed(0)
    layer = MagicMock()
    layer.swa_cache_layer.kv_cache = _make_cache(WINDOW, CACHE_BLOCK_SIZE)
    layer.scale = HEAD_DIM**-0.5
    layer.attn_sink = torch.randn(num_heads, dtype=torch.float32, device="cuda")
    layer.head_dim = HEAD_DIM
    layer.nope_head_dim = NOPE_HEAD_DIM
    layer.rope_head_dim = ROPE_HEAD_DIM
    swa_metadata = _swa_metadata()
    if case == "swa_only":
        layer.compress_ratio = 1
        attn_metadata, kv_cache = None, None
    elif case == "c4":
        attn_metadata, kv_cache = _c4_case(layer)
    else:
        attn_metadata, kv_cache = _c128_case(layer)

    q = torch.randn(
        (NUM_TOKENS, num_heads, HEAD_DIM), dtype=torch.float16, device="cuda"
    )
    workspace_manager = MagicMock()
    workspace_manager.get_simultaneous.side_effect = lambda *specs: tuple(
        torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in specs
    )

    outputs = {}
    for enabled in (False, True):
        output = torch.empty_like(q)
        with (
            patch.object(rocm, "_qk_dsplit_decode_enabled", return_value=enabled),
            patch.object(
                rocm, "current_workspace_manager", return_value=workspace_manager
            ),
        ):
            rocm.DeepseekV4ROCMAiterMLASparseImpl._forward_decode(
                layer,
                q,
                kv_cache,
                swa_metadata,
                attn_metadata,
                case == "swa_only",
                output,
            )
        outputs[enabled] = output

    assert workspace_manager.get_simultaneous.call_count == 1
    assert torch.isfinite(outputs[True]).all()
    torch.testing.assert_close(outputs[True], outputs[False], atol=2e-3, rtol=2e-3)
