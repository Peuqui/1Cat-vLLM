# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared next-token derivation for padded speculative sampler output.

`prepare_next_token_ids_padded` wraps eagle_prepare_next_token_padded_kernel
and is used by the drafter on the last PP rank and by the non-last PP ranks
on the sampled matrix they receive from it, so both derive identical values.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.spec_decode.utils import prepare_next_token_ids_padded


def reference_prepare_next_token_ids_padded(
    sampled_token_ids: torch.Tensor,
    discard_request_mask: torch.Tensor,
    backup_next_token_ids: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch mirror of eagle_prepare_next_token_padded_kernel.

    Only the contiguous valid prefix of a row counts; the next token is its
    last token, or the backup token for discarded rows and rows without a
    valid token.
    """
    valid = (sampled_token_ids >= 0) & (sampled_token_ids < vocab_size)
    counts = valid.to(torch.int32).cumprod(dim=1).sum(dim=1).to(torch.int32)
    last = sampled_token_ids.gather(
        1, (counts.to(torch.int64) - 1).clamp_(min=0).unsqueeze(1)
    ).squeeze(1)
    num_reqs = sampled_token_ids.shape[0]
    backup = backup_next_token_ids[:num_reqs]
    next_token_ids = torch.where(counts > 0, last.to(torch.int32), backup)
    discarded = discard_request_mask[:num_reqs]
    next_token_ids = torch.where(discarded, backup, next_token_ids)
    counts = torch.where(discarded, torch.zeros_like(counts), counts)
    return next_token_ids, counts


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel needs a CUDA device"
)
@pytest.mark.parametrize("num_spec_tokens", [1, 3, 7])
def test_prepare_next_token_ids_padded_matches_reference(num_spec_tokens: int):
    device = torch.device(current_platform.device_type)
    vocab_size = 50
    num_reqs = 64
    generator = torch.Generator(device="cpu").manual_seed(num_spec_tokens)
    sampled = torch.randint(
        0, vocab_size, (num_reqs, num_spec_tokens + 1), generator=generator
    )
    # Rejection padding, then stale values after the first -1 that must not
    # count, and an out-of-vocab value that ends the valid prefix as well.
    first_invalid = torch.randint(
        0, num_spec_tokens + 2, (num_reqs,), generator=generator
    )
    for row, col in enumerate(first_invalid.tolist()):
        if col <= num_spec_tokens:
            sampled[row, col] = -1
            sampled[row, col + 1 :] = 7
    sampled[0, 0] = -1  # no valid token at all
    sampled[1, 0] = vocab_size  # out of vocab in the first slot
    discard = torch.zeros(num_reqs, dtype=torch.bool)
    discard[::5] = True
    backup = torch.arange(1000, 1000 + num_reqs, dtype=torch.int32)

    expected_ids, expected_counts = reference_prepare_next_token_ids_padded(
        sampled, discard, backup, vocab_size
    )
    next_token_ids, counts = prepare_next_token_ids_padded(
        sampled.to(device, torch.int32),
        discard.to(device),
        backup.to(device),
        vocab_size,
    )
    assert next_token_ids.cpu().tolist() == expected_ids.tolist()
    assert counts.cpu().tolist() == expected_counts.tolist()
    assert next_token_ids.dtype == torch.int32
    assert counts.dtype == torch.int32
