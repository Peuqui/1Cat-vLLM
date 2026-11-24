# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host block-table gather runs only for builders that read it."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAMetadataBuilder
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _prepare(needs_cpu: bool):
    calls: list[str] = []
    tables = SimpleNamespace(
        gather_block_tables=lambda *a, **k: ("gpu",),
        gather_block_tables_cpu=lambda *a, **k: calls.append("cpu") or ("cpu",),
        compute_slot_mappings=lambda *a, **k: "slots",
    )
    runner = SimpleNamespace(block_tables=tables, _needs_block_table_cpu=needs_cpu)
    batch = SimpleNamespace(
        idx_mapping=torch.zeros(1, dtype=torch.int32),
        idx_mapping_np=np.zeros(1, dtype=np.int32),
        num_reqs_after_padding=1,
        query_start_loc=None,
        positions=None,
        num_tokens_after_padding=1,
    )
    GPUModelRunner.prepare_attn(runner, batch)
    return calls, batch


def test_no_reader_skips_host_gather() -> None:
    calls, batch = _prepare(needs_cpu=False)
    assert calls == []
    assert not hasattr(batch, "_block_tables_cpu")


def test_reader_gets_the_host_gather() -> None:
    calls, batch = _prepare(needs_cpu=True)
    assert calls == ["cpu"]
    assert batch._block_tables_cpu == ("cpu",)


@pytest.mark.parametrize(
    "builder,expected",
    [(AttentionMetadataBuilder, False), (Qwen4ExpQSAMetadataBuilder, True)],
)
def test_only_qsa_builder_declares_the_host_table(builder, expected) -> None:
    assert builder.requires_block_table_cpu is expected
