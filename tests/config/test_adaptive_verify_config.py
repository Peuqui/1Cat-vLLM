# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.kernel import (
    AdaptiveVerifyConfig,
    KernelConfig,
    resolve_adaptive_verify,
)

pytestmark = pytest.mark.cpu_test

DFLASH = SimpleNamespace(method="dflash")


def _resolve(method_config=DFLASH, k_max: int = 7, **adaptive_verify):
    kernel = KernelConfig(adaptive_verify=AdaptiveVerifyConfig(**adaptive_verify))
    return resolve_adaptive_verify(method_config, kernel, k_max)


def test_adaptive_verify_does_not_change_the_compile_hash():
    assert (
        KernelConfig().compute_hash()
        == KernelConfig(
            adaptive_verify=AdaptiveVerifyConfig(
                enabled=True, forced_tokens=3, switch_margin=0.5
            )
        ).compute_hash()
    )


def test_disabled_is_fixed_length_for_dflash():
    assert _resolve() is None


def test_defaults_span_one_to_the_drafter_block():
    plan = _resolve(enabled=True)
    assert (plan.rule, plan.min_tokens, plan.max_tokens, plan.initial_tokens) == (
        "throughput",
        1,
        7,
        7,
    )


def test_explicit_bounds_and_first_round():
    plan = _resolve(enabled=True, min_tokens=2, max_tokens=6, initial_tokens=4)
    assert (plan.min_tokens, plan.max_tokens, plan.initial_tokens) == (2, 6, 4)


def test_forced_tokens_select_the_calibration_rule():
    plan = _resolve(enabled=True, forced_tokens=5)
    assert (plan.rule, plan.initial_tokens) == ("forced", 5)


@pytest.mark.parametrize(
    "settings",
    [
        {"min_tokens": 5, "max_tokens": 3},
        {"max_tokens": 8},
        {"initial_tokens": 8},
        {"min_tokens": 3, "initial_tokens": 2},
        {"forced_tokens": 8},
        {"min_tokens": 3, "forced_tokens": 2},
    ],
)
def test_inconsistent_bounds_are_rejected(settings):
    with pytest.raises(ValueError, match="adaptive_verify"):
        _resolve(enabled=True, **settings)


@pytest.mark.parametrize("method", ["eagle", "ngram", "medusa", None])
def test_enabling_it_for_another_method_is_an_error(method):
    with pytest.raises(ValueError, match="requires speculative method 'dflash'"):
        _resolve(SimpleNamespace(method=method), enabled=True)


def test_enabling_it_for_mtp_is_an_error():
    with pytest.raises(ValueError, match="does not apply to MTP"):
        _resolve(SimpleNamespace(method="mtp"), enabled=True)


@pytest.mark.parametrize("field", ["ema_alpha", "prior_weight", "verify_cost_ratio"])
def test_controller_parameters_must_be_positive(field):
    with pytest.raises(ValueError):
        AdaptiveVerifyConfig(**{field: 0})
