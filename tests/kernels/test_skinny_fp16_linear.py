# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny FP16 linear for decode steps (vllm/model_executor/kernels/linear/
skinny_fp16_linear.py), on the production shapes of DeepSeek-V4-Flash and
GLM-5.3-Flash listed in benchmarks/kernels/benchmark_skinny_fp16_linear.py."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from benchmarks.kernels.benchmark_skinny_fp16_linear import SHAPES, Shape
from tests.kernels.utils import requires_tensor_core_triton
from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels.linear import skinny_fp16_linear as skinny_module
from vllm.model_executor.kernels.linear.skinny_fp16_linear import (
    SKINNY_FP16_LINEAR_MAX_CHUNKS,
    SKINNY_FP16_LINEAR_MAX_ROWS,
    SKINNY_FP16_LINEAR_MAX_SPLIT_K,
    SKINNY_FP16_LINEAR_MAX_WEIGHT_BYTES,
    SkinnyTile,
    skinny_fp16_linear,
    skinny_fp16_linear_apply,
    skinny_fp16_linear_decision,
    skinny_fp16_linear_device_traits,
    skinny_fp16_linear_layer_decision,
    skinny_fp16_linear_tile,
)
from vllm.model_executor.layers import linear as linear_module
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import mma_probe
from vllm.utils.torch_utils import set_high_precision_cuda_matmul_defaults

ROWS = [1, 2, 4, 5, 6, 7, 8]
CARDS = [(70, 80), (75, 72)]  # (capability, SM count): V100, Quadro RTX 8000
# FP16 rounding of a result is at most 2^-11 of it, which is at most 2^-11 of
# sum |x||w|; the FP32 accumulation adds about sqrt(K) * 2^-24 of that sum.
FP16_ERROR_BOUND = 2**-11 + 1e-6
# FP32 output has no rounding of its own: only the accumulation error remains,
# about 4e-6 of sum |x||w| at K = 4096 and 8e-6 at 8192; 5e-5 leaves 6x margin
# and is still 10x below the FP16 bound.
FP32_ERROR_BOUND = 5e-5


@pytest.fixture(autouse=True)
def worker_matmul_policy():
    """The numerical policy a worker sets before loading (gpu_worker.py): no
    reduced-precision FP16 reductions, so the cuBLAS arm computes as in serving.
    The flags are restored afterwards; the split-K half is read-only and is
    written back together with its reduction flag as a tuple."""
    matmul = torch.backends.cuda.matmul
    saved: dict[str, object] = {}
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        split_k_name = f"{name}_split_k"
        saved[name] = (
            (getattr(matmul, name), getattr(matmul, split_k_name))
            if hasattr(matmul, split_k_name)
            else getattr(matmul, name)
        )
    if hasattr(matmul, "allow_fp16_accumulation"):
        saved["allow_fp16_accumulation"] = matmul.allow_fp16_accumulation
    set_high_precision_cuda_matmul_defaults()
    yield
    for name, value in saved.items():
        setattr(matmul, name, value)


# --- host tests: coverage and tile choice -----------------------------------


def _decide(x, w, bias=None, out_dtype=torch.float16, card=CARDS[0]):
    return skinny_fp16_linear_decision(
        x, w, bias, out_dtype, capability=card[0], sm_count=card[1]
    )


def _host(rows=5, n=256, k=4096):
    return torch.empty(rows, k, dtype=torch.half), torch.empty(n, k, dtype=torch.half)


@pytest.mark.parametrize("card", CARDS, ids=lambda card: f"sm{card[0]}")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: shape.name)
def test_production_shapes_are_covered_with_a_sound_tile(shape: Shape, card):
    x, w = _host(rows=5, n=shape.n, k=shape.k)
    decision = _decide(x, w, out_dtype=shape.out_dtype, card=card)
    if shape.n * shape.k * 2 > SKINNY_FP16_LINEAR_MAX_WEIGHT_BYTES:
        assert decision.tile is None and "MiB" in decision.reason
        return
    assert decision.tile is not None, decision.reason
    tile = decision.tile
    chunks = shape.k // (tile.block_k * tile.split_k)
    # The tile divides K exactly, no program unrolls more than the cap, the
    # split is a power of two within the cap.
    assert shape.k % (tile.block_k * tile.split_k) == 0
    assert 1 <= chunks <= SKINNY_FP16_LINEAR_MAX_CHUNKS
    assert tile.split_k & (tile.split_k - 1) == 0
    assert tile.split_k <= SKINNY_FP16_LINEAR_MAX_SPLIT_K
    # Every SM gets a program wherever K allows it.
    programs = -(-shape.n // tile.block_n) * tile.split_k
    assert (
        programs >= card[1]
        or tile.split_k == SKINNY_FP16_LINEAR_MAX_SPLIT_K
        or chunks == 1
    )


# The tile the rule gives the shapes of the tile sweep of 25.11.2026 (the rule is
# fitted to that sweep; benchmark_skinny_fp16_linear.py --sweep checks it again).
# (N, K): (block_n, block_k, num_warps, split_k)
RULE_TILES = {
    70: {
        (256, 4096): (32, 512, 4, 8),
        (2048, 4096): (32, 128, 2, 4),
        (1024, 4096): (32, 128, 2, 4),
        (512, 4096): (32, 128, 2, 8),
        (64, 4096): (32, 512, 4, 8),
        (288, 4096): (32, 512, 4, 8),
        (320, 4096): (32, 128, 2, 8),
        (160, 4096): (32, 512, 4, 8),
        (128, 4096): (32, 512, 4, 8),
        (4096, 1536): (32, 128, 2, 2),
    },
    75: {
        (256, 4096): (16, 256, 2, 4),
        (2048, 4096): (32, 256, 4, 2),
        (1024, 4096): (32, 256, 4, 2),
        (512, 4096): (32, 256, 4, 4),
        (64, 4096): (16, 256, 2, 16),
        (288, 4096): (32, 256, 4, 4),
        (320, 4096): (32, 256, 4, 4),
        (160, 4096): (16, 256, 2, 8),
        (128, 4096): (16, 256, 2, 8),
        (4096, 1536): (32, 256, 4, 1),
    },
}


@pytest.mark.parametrize("card", CARDS, ids=lambda card: f"sm{card[0]}")
def test_rule_gives_the_swept_tiles(card):
    for (n, k), expected in RULE_TILES[card[0]].items():
        assert skinny_fp16_linear_tile(n, k, *card).tile == SkinnyTile(*expected)


def test_a_split_is_taken_only_when_the_columns_cannot_fill_the_card():
    # 256 columns are 16 strips of 16: not enough programs, so K is split.
    assert skinny_fp16_linear_tile(256, 4096, 70, 80).tile.split_k > 1
    # 8192 columns of a short K fill the card alone.
    assert skinny_fp16_linear_tile(8192, 128, 70, 80).tile.split_k == 1


def test_unknown_capability_has_no_tile():
    decision = skinny_fp16_linear_tile(256, 4096, 80, 108)
    assert decision.tile is None and "capability 80" in decision.reason


def test_k_too_long_to_unroll_is_refused_with_the_reason():
    # 2^20 / 128 chunks cannot be cut into at most 8 per program within split 32.
    decision = skinny_fp16_linear_tile(256, 2**20, 70, 80)
    assert decision.tile is None and "cannot be split" in decision.reason


@pytest.mark.parametrize("rows", ROWS)
def test_covers_one_to_eight_rows(rows):
    x, w = _host(rows=rows)
    assert _decide(x, w).tile is not None


@pytest.mark.parametrize(
    "x_shape, w_shape, x_dtype, w_dtype, reason",
    [
        ((SKINNY_FP16_LINEAR_MAX_ROWS + 1, 4096), (256, 4096), None, None, "1..8 rows"),
        ((0, 4096), (256, 4096), None, None, "1..8 rows"),
        ((5, 4096), (256, 2048), None, None, "K mismatch"),
        ((5, 4104), (256, 4104), None, None, "multiple of 16"),
        ((5, 4096), (256, 4096), torch.bfloat16, None, "FP16 x"),
        ((5, 4096), (256, 4096), None, torch.float32, "FP16 weight"),
        ((5, 4096), (256, 4096), None, torch.float8_e4m3fn, "FP16 weight"),
        ((2, 5, 4096), (256, 4096), None, None, r"x \[M, K\]"),
        ((5, 4096), (4096,), None, None, r"weight \[N, K\]"),
    ],
)
def test_rejects_what_does_not_fit(x_shape, w_shape, x_dtype, w_dtype, reason):
    x = torch.empty(x_shape, dtype=x_dtype or torch.half)
    w = torch.empty(w_shape, dtype=w_dtype or torch.half)
    decision = _decide(x, w)
    assert decision.tile is None
    assert decision.reason and reason.replace("\\", "") in decision.reason.replace(
        "\\", ""
    )


def test_rejects_strides_and_devices_the_kernel_cannot_read():
    x, w = _host()
    # Transposed weight ([K, N] storage): K is not the contiguous axis.
    assert "contiguous" in _decide(x, w.T.contiguous().T).reason
    # Strided activations along K.
    assert "contiguous" in _decide(x.T.contiguous().T, w).reason
    # Overlapping weight rows.
    overlapping = torch.empty(4096 + 256, dtype=torch.half).as_strided(
        (256, 4096), (1, 1)
    )
    assert "overlap" in _decide(x, overlapping).reason
    # Another device (meta stands in for a second card).
    assert "weight on meta" in _decide(x, w.to("meta")).reason


def test_accepts_row_slices_of_a_merged_weight():
    x, w = _host()
    merged = torch.empty(256 + 64, 4096 + 64, dtype=torch.half)
    sliced = merged[16 : 16 + 256, :4096]  # row and column slice, row stride 4160
    assert sliced.stride() == (4160, 1)
    assert _decide(x, sliced).tile is not None


def test_bias_and_output_dtype_coverage():
    x, w = _host()
    bias = torch.empty(256, dtype=torch.half)
    assert _decide(x, w, bias).tile is not None
    assert _decide(x, w, out_dtype=torch.float32).tile is not None
    assert "bias" in _decide(x, w, bias.float()).reason
    assert "bias" in _decide(x, w, torch.empty(255, dtype=torch.half)).reason
    assert "bias" in _decide(x, w, torch.empty(512, dtype=torch.half)[::2]).reason
    assert "output" in _decide(x, w, out_dtype=torch.bfloat16).reason


def test_launcher_refuses_what_the_decision_refuses_and_a_foreign_tile():
    x, w = _host()
    out = torch.empty(5, 256, dtype=torch.half)
    tile = SkinnyTile(16, 128, 4, 1)
    with pytest.raises(ValueError, match="1..8 rows"):
        skinny_fp16_linear(torch.empty(9, 4096, dtype=torch.half), w, out, tile)
    with pytest.raises(ValueError, match="K=4096 is not a multiple"):
        skinny_fp16_linear(x, w, out, SkinnyTile(16, 128, 4, 3))
    with pytest.raises(ValueError, match="out must be"):
        skinny_fp16_linear(x, w, torch.empty(5, 255, dtype=torch.half), tile)


# --- GPU tests --------------------------------------------------------------


def _traits():
    return skinny_fp16_linear_device_traits(
        torch.device("cuda", torch.cuda.current_device())
    )


def _tile(x, w, bias=None, out_dtype=torch.float16) -> SkinnyTile:
    """The rule's tile, also for the weights the decision leaves to cuBLAS (the
    correctness of the kernel does not depend on that policy)."""
    tile = skinny_fp16_linear_tile(w.shape[0], w.shape[1], *_traits()).tile
    assert tile is not None
    return tile


def _inputs(rows, n, k, seed, bias=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, k, generator=gen, device="cuda").half()
    w = (torch.randn(n, k, generator=gen, device="cuda") * 0.02).half()
    b = (torch.randn(n, generator=gen, device="cuda") * 0.5).half() if bias else None
    return x, w, b


def _relative_error(out, x, w, bias=None):
    """Error against an FP64 product, relative to sum |x||w| (the rounding of a
    result to FP16 alone is up to 2^-11 of that; a drifting accumulation is not)."""
    xd, wd = x.double(), w.double()
    reference = xd @ wd.T
    scale = xd.abs() @ wd.abs().T
    if bias is not None:
        reference += bias.double()
        scale += bias.double().abs()
    return (out.double() - reference).abs() / scale


@requires_tensor_core_triton
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: shape.name)
def test_matches_fp64_at_least_as_well_as_cublas(shape: Shape, rows):
    x, w, _ = _inputs(rows, shape.n, shape.k, seed=rows)
    out = torch.empty(rows, shape.n, dtype=shape.out_dtype, device="cuda")
    skinny_fp16_linear(x, w, out, _tile(x, w, out_dtype=shape.out_dtype))
    cublas = (
        torch.mm(x, w.T, out_dtype=torch.float32)
        if shape.out_dtype == torch.float32
        else F.linear(x, w)
    )

    error_skinny = _relative_error(out, x, w)
    error_cublas = _relative_error(cublas, x, w)
    if shape.out_dtype == torch.float16:
        assert error_skinny.max() <= FP16_ERROR_BOUND
        # No worse than the kernel it replaces, in the mean and at the worst
        # (both are the same accumulation rounded once; they differ only where a
        # result sits at a rounding boundary).
        assert error_skinny.mean() <= error_cublas.mean() * 1.05
        assert error_skinny.max() <= max(error_cublas.max() * 1.05, 2**-12)
    else:
        assert error_skinny.max() <= FP32_ERROR_BOUND
    # Sanity of the oracle itself: the two products agree to FP16 rounding.
    torch.testing.assert_close(out, cublas, rtol=2e-3, atol=2e-3)


@requires_tensor_core_triton
@pytest.mark.parametrize("rows", [1, 5, 8])
@pytest.mark.parametrize(
    "tile",
    [SkinnyTile(16, 128, 4, 1), SkinnyTile(16, 64, 4, 8), SkinnyTile(32, 256, 8, 2)],
    ids=str,
)
def test_bias_is_added_before_the_single_rounding(rows, tile):
    n, k = 288, 4096
    x, w, bias = _inputs(rows, n, k, seed=3, bias=True)
    out = torch.empty(rows, n, dtype=torch.half, device="cuda")
    skinny_fp16_linear(x, w, out, tile, bias)
    assert _relative_error(out, x, w, bias).max() <= FP16_ERROR_BOUND
    torch.testing.assert_close(out, F.linear(x, w, bias), rtol=2e-3, atol=2e-3)
    # The bias is part of the fixed-order sum: dropping it changes the result.
    no_bias = torch.empty_like(out)
    skinny_fp16_linear(x, w, no_bias, tile)
    assert not torch.equal(out, no_bias)


@requires_tensor_core_triton
@pytest.mark.parametrize("tile", [SkinnyTile(16, 128, 4, 1), SkinnyTile(16, 128, 4, 8)])
def test_fp32_output_is_the_accumulator_rounded_once_to_fp16(tile):
    x, w, _ = _inputs(6, 256, 4096, seed=4)
    out16 = torch.empty(6, 256, dtype=torch.half, device="cuda")
    out32 = torch.empty(6, 256, dtype=torch.float32, device="cuda")
    skinny_fp16_linear(x, w, out16, tile)
    skinny_fp16_linear(x, w, out32, tile)
    assert torch.equal(out16, out32.half())
    assert _relative_error(out32, x, w).max() <= FP32_ERROR_BOUND
    # The FP32 result keeps what FP16 rounding drops.
    assert not torch.equal(out32, out32.half().float())


@requires_tensor_core_triton
def test_tiles_agree_up_to_summation_order():
    x, w, _ = _inputs(6, 256, 4096, seed=2)
    default = torch.empty(6, 256, dtype=torch.half, device="cuda")
    skinny_fp16_linear(x, w, default, _tile(x, w))
    for tile in (
        SkinnyTile(16, 128, 4, 1),
        SkinnyTile(16, 512, 4, 2),
        SkinnyTile(32, 64, 2, 16),
        SkinnyTile(64, 256, 8, 4),
    ):
        other = torch.empty_like(default)
        skinny_fp16_linear(x, w, other, tile)
        torch.testing.assert_close(other, default, rtol=2e-3, atol=2e-3)


@requires_tensor_core_triton
@pytest.mark.parametrize("split_k", [1, 8])
def test_deterministic_between_calls(split_k):
    x, w, _ = _inputs(7, 288, 4096, seed=1)
    tile = SkinnyTile(16, 128, 4, split_k)
    first = torch.empty(7, 288, dtype=torch.half, device="cuda")
    second = torch.empty_like(first)
    skinny_fp16_linear(x, w, first, tile)
    skinny_fp16_linear(x, w, second, tile)
    assert torch.equal(first, second)


@requires_tensor_core_triton
@pytest.mark.parametrize("split_k", [1, 4])
def test_writes_only_its_rows_and_columns(split_k):
    # N = 100 is not a multiple of the 16-wide strip; the output is a view into
    # a larger buffer with its own row stride.
    rows, n, k = 5, 100, 512
    x, w, _ = _inputs(rows, n, k, seed=0)
    buffer = torch.full((16, n + 28), -7.0, dtype=torch.half, device="cuda")
    skinny_fp16_linear(x, w, buffer[:rows, :n], SkinnyTile(16, 128, 4, split_k))
    assert (buffer[rows:] == -7.0).all()
    assert (buffer[:, n:] == -7.0).all()
    assert (buffer[:rows, :n] != -7.0).all()
    assert _relative_error(buffer[:rows, :n], x, w).max() <= FP16_ERROR_BOUND


@requires_tensor_core_triton
def test_strided_operands_give_the_contiguous_result():
    rows, n, k = 6, 256, 4096
    x, w, _ = _inputs(rows, n, k, seed=5)
    tile = _tile(x, w)
    reference = torch.empty(rows, n, dtype=torch.half, device="cuda")
    skinny_fp16_linear(x, w, reference, tile)

    merged = torch.zeros(n + 48, k + 64, dtype=torch.half, device="cuda")
    merged[16 : 16 + n, :k] = w  # row slice of a wider merged weight
    wide_x = torch.zeros(rows, k + 32, dtype=torch.half, device="cuda")
    wide_x[:, :k] = x
    out = torch.empty(rows, n, dtype=torch.half, device="cuda")
    skinny_fp16_linear(wide_x[:, :k], merged[16 : 16 + n, :k], out, tile)
    assert torch.equal(out, reference)


@requires_tensor_core_triton
@pytest.mark.parametrize("split_k", [1, 8])
def test_cuda_graph_replays_with_new_inputs(split_k):
    rows, n, k = 5, 256, 4096
    x, w, bias = _inputs(rows, n, k, seed=6, bias=True)
    tile = SkinnyTile(16, 128, 4, split_k)
    out = torch.empty(rows, n, dtype=torch.half, device="cuda")
    skinny_fp16_linear(x, w, out, tile, bias)  # compile outside the capture
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):  # the workspace is allocated inside the capture
            skinny_fp16_linear(x, w, out, tile, bias)
    for seed in (7, 8):
        fresh, _, _ = _inputs(rows, n, k, seed=seed)
        x.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.empty_like(out)
        skinny_fp16_linear(x, w, expected, tile, bias)
        assert torch.equal(out, expected)
        assert _relative_error(out, x, w, bias).max() <= FP16_ERROR_BOUND


# --- host tests: the weight limit and the config switch ----------------------


@pytest.mark.parametrize("card", CARDS, ids=lambda card: f"sm{card[0]}")
def test_weights_above_the_measured_win_stay_on_cublas(card):
    def decide(n, k):
        weight = torch.empty(n, k, dtype=torch.half, device="meta")
        return skinny_fp16_linear_layer_decision(
            weight, None, torch.float16, capability=card[0], sm_count=card[1]
        )

    # Largest weight the kernel beat cuBLAS on: the 16.8 MB compressor.
    assert decide(2048, 4096).tile is not None
    # The 66 MB DSpark markov head and the 67 MB GLM eh_proj.
    for n, k in ((129280, 256), (4096, 8192)):
        decision = decide(n, k)
        assert decision.tile is None and "MiB" in decision.reason


def test_layer_decision_needs_no_activation_and_checks_the_bias():
    weight = torch.empty(256, 4096, dtype=torch.half, device="meta")
    bias = torch.empty(256, dtype=torch.half, device="meta")
    assert (
        skinny_fp16_linear_layer_decision(
            weight, bias, torch.float16, capability=70, sm_count=80
        ).tile
        is not None
    )
    reason = skinny_fp16_linear_layer_decision(
        weight, bias.float(), torch.float16, capability=70, sm_count=80
    ).reason
    assert "bias" in reason


DEVICE_INDEX = 3


def _sm7x_device(monkeypatch, capability: int) -> None:
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=None: DeviceCapability(capability // 10, capability % 10),
    )
    monkeypatch.setattr(mma_probe, "lowers_dot_to_mma", lambda capability: True)


def test_switch_is_off_by_default():
    assert not KernelConfig().use_skinny_fp16_linear(DEVICE_INDEX)


def test_switch_needs_the_tensor_core_triton_setting():
    with pytest.raises(ValueError, match="triton_sm7x_tensor_cores"):
        KernelConfig(skinny_fp16_linear=True).use_skinny_fp16_linear(DEVICE_INDEX)


@pytest.mark.parametrize("capability, expected", [(70, True), (75, True), (80, False)])
def test_switch_is_on_for_sm7x_cards(monkeypatch, capability, expected):
    _sm7x_device(monkeypatch, capability)
    config = KernelConfig(skinny_fp16_linear=True, triton_sm7x_tensor_cores=True)
    assert config.use_skinny_fp16_linear(DEVICE_INDEX) is expected


def test_switch_without_a_tensor_core_triton_stops_the_engine(monkeypatch):
    _sm7x_device(monkeypatch, 70)
    monkeypatch.setattr(mma_probe, "lowers_dot_to_mma", lambda capability: False)
    config = KernelConfig(skinny_fp16_linear=True, triton_sm7x_tensor_cores=True)
    with pytest.raises(RuntimeError, match="lowers tl.dot to FMA"):
        config.use_skinny_fp16_linear(DEVICE_INDEX)


def test_switch_is_part_of_the_compilation_hash():
    # The layers run inside compiled graphs: the graph differs with the switch.
    on = KernelConfig(skinny_fp16_linear=True, triton_sm7x_tensor_cores=True)
    off = KernelConfig(triton_sm7x_tensor_cores=True)
    assert on.compute_hash() != off.compute_hash()


# --- host tests: the opaque op and the layer method --------------------------


@pytest.mark.parametrize("rows", [0, 1, 8, 9, 512])
@pytest.mark.parametrize("float32_output", [False, True])
def test_op_takes_the_kernel_for_one_to_eight_fp16_rows_only(
    monkeypatch, rows, float32_output
):
    calls: list[str] = []
    monkeypatch.setattr(
        skinny_module, "skinny_fp16_linear", lambda *args: calls.append("kernel")
    )
    monkeypatch.setattr(
        F, "linear", lambda x, w, b: calls.append("cublas") or x.new_empty(0)
    )
    monkeypatch.setattr(
        torch, "mm", lambda x, w, out_dtype: calls.append("cublas") or x.new_empty(0)
    )
    x = torch.empty(rows, 64, dtype=torch.half)
    w = torch.empty(32, 64, dtype=torch.half)
    skinny_module._skinny_fp16_linear_op(x, w, None, 16, 64, 4, 1, float32_output)
    assert calls == ["kernel" if 1 <= rows <= 8 else "cublas"]


def test_op_keeps_cublas_for_other_dtypes_and_strides(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        skinny_module, "skinny_fp16_linear", lambda *args: calls.append("kernel")
    )
    monkeypatch.setattr(
        F, "linear", lambda x, w, b: calls.append("cublas") or x.new_empty(0)
    )
    w = torch.empty(32, 64, dtype=torch.half)
    skinny_module._skinny_fp16_linear_op(
        torch.empty(5, 64, dtype=torch.float32), w, None, 16, 64, 4, 1, False
    )
    skinny_module._skinny_fp16_linear_op(
        torch.empty(64, 5, dtype=torch.half).T, w, None, 16, 64, 4, 1, False
    )
    assert calls == ["cublas", "cublas"]


def test_op_fake_allocates_the_output_like_the_real_op():
    x = torch.empty(5, 64, dtype=torch.half, device="meta")
    w = torch.empty(32, 64, dtype=torch.half, device="meta")
    for float32_output, dtype in ((False, torch.half), (True, torch.float32)):
        out = skinny_module._skinny_fp16_linear_op_fake(
            x, w, None, 16, 64, 4, 1, float32_output
        )
        assert out.shape == (5, 32) and out.dtype == dtype


def _layer(tile):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty(256, 4096, dtype=torch.half), requires_grad=False
    )
    layer.skinny_tile = tile
    layer._layer_trace = SimpleNamespace(unquant_debug=False)
    return layer


def test_unquantized_method_takes_the_op_for_a_layer_the_decision_admitted(
    monkeypatch,
):
    calls: list[str] = []
    tile = SkinnyTile(32, 512, 4, 8)

    def apply(x, weight, bias, got_tile, **kwargs):
        calls.append("skinny")
        assert got_tile == tile and not kwargs
        return x.new_empty(x.shape[0], weight.shape[0])

    monkeypatch.setattr(skinny_module, "skinny_fp16_linear_apply", apply)
    monkeypatch.setattr(
        linear_module,
        "dispatch_unquantized_gemm",
        lambda: lambda layer, x, w, b: calls.append("cublas") or x.new_empty(0),
    )
    method = UnquantizedLinearMethod()
    x = torch.empty(5, 4096, dtype=torch.half)
    method.apply(_layer(tile), x)
    method.apply(_layer(None), x)
    # Batches above 8 rows reach the op too: it keeps cuBLAS for them itself,
    # because a compiled graph cannot branch on the batch.
    method.apply(_layer(tile), torch.empty(64, 4096, dtype=torch.half))
    assert calls == ["skinny", "cublas", "skinny"]


def test_unquantized_method_keeps_batch_dimensions(monkeypatch):
    monkeypatch.setattr(
        skinny_module,
        "skinny_fp16_linear_apply",
        lambda x, weight, bias, tile, **kwargs: x.new_empty(
            x.shape[0], weight.shape[0]
        ),
    )
    out = UnquantizedLinearMethod().apply(
        _layer(SkinnyTile(32, 512, 4, 8)), torch.empty(2, 3, 4096, dtype=torch.half)
    )
    assert out.shape == (2, 3, 256)


def test_prepare_leaves_a_weight_that_is_not_on_cuda_on_cublas():
    layer = _layer(SkinnyTile(32, 512, 4, 8))
    layer.weight = torch.nn.Parameter(
        torch.empty(256, 4096, dtype=torch.half, device="meta"), requires_grad=False
    )
    linear_module._prepare_skinny_fp16_linear(layer)
    assert layer.skinny_tile is None


# --- host tests: GateLinear ---------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA platform")
@pytest.mark.parametrize("widen_in_router", [False, True])
def test_gate_leaves_the_widening_to_the_router_when_asked(
    default_vllm_config, widen_in_router
):
    gate = GateLinear(
        4096,
        256,
        out_dtype=torch.float32,
        params_dtype=torch.half,
        widen_in_router=widen_in_router,
        prefix="gate",
    )
    torch.nn.init.normal_(gate.weight, std=0.02)
    x = torch.randn(5, 4096).half()
    logits, _ = gate(x)
    expected = F.linear(x, gate.weight)
    assert logits.dtype == (torch.half if widen_in_router else torch.float32)
    # Either way the values are the FP16 product; widening is exact.
    assert torch.equal(logits.float(), expected.float())


# --- GPU tests: layer, compile and graph --------------------------------------


@requires_tensor_core_triton
def test_layer_with_the_switch_takes_the_kernel_for_few_rows(monkeypatch):
    n, k = 256, 4096
    _, w, bias = _inputs(1, n, k, seed=11, bias=True)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    on = SimpleNamespace(
        kernel_config=KernelConfig(
            skinny_fp16_linear=True, triton_sm7x_tensor_cores=True
        )
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: on)
    linear_module._prepare_skinny_fp16_linear(layer)
    tile = layer.skinny_tile
    assert tile == skinny_fp16_linear_tile(n, k, *_traits()).tile

    method = UnquantizedLinearMethod()
    x, _, _ = _inputs(9, n, k, seed=12)
    for rows in (5, 8):
        direct = torch.empty(rows, n, dtype=torch.half, device="cuda")
        skinny_fp16_linear(x[:rows], w, direct, tile, bias)
        assert torch.equal(method.apply(layer, x[:rows], bias), direct)
    # Nine rows: the call the layer made before.
    assert torch.equal(method.apply(layer, x, bias), F.linear(x, w, bias))

    off = SimpleNamespace(kernel_config=KernelConfig())
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: off)
    linear_module._prepare_skinny_fp16_linear(layer)
    assert layer.skinny_tile is None


@requires_tensor_core_triton
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("float32_output", [False, True])
def test_op_under_fullgraph_compile_matches_the_direct_call(backend, float32_output):
    n, k = 256, 4096
    x, w, bias = _inputs(9, n, k, seed=13, bias=not float32_output)
    tile = skinny_fp16_linear_tile(n, k, *_traits()).tile
    assert tile is not None

    def product(rows):
        if rows > SKINNY_FP16_LINEAR_MAX_ROWS:
            return (
                torch.mm(x[:rows], w.T, out_dtype=torch.float32)
                if float32_output
                else F.linear(x[:rows], w, bias)
            )
        out = torch.empty(
            rows,
            n,
            dtype=torch.float32 if float32_output else torch.half,
            device="cuda",
        )
        skinny_fp16_linear(x[:rows], w, out, tile, bias)
        return out

    compiled = torch.compile(
        lambda xs: skinny_fp16_linear_apply(
            xs, w, bias, tile, float32_output=float32_output
        ),
        fullgraph=True,
        dynamic=True,
        backend=backend,
    )
    # One traced graph serves every batch: the branch is inside the op.
    for rows in (5, 8, 9, 2):
        assert torch.equal(compiled(x[:rows]), product(rows))


@requires_tensor_core_triton
@pytest.mark.parametrize("float32_output", [False, True])
def test_op_in_a_cuda_graph_replays_with_new_inputs(float32_output):
    n, k = 288, 4096
    x, w, bias = _inputs(6, n, k, seed=14, bias=not float32_output)
    tile = skinny_fp16_linear_tile(n, k, *_traits()).tile
    assert tile is not None
    bias = None if float32_output else bias
    skinny_fp16_linear_apply(x, w, bias, tile, float32_output=float32_output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = skinny_fp16_linear_apply(
            x, w, bias, tile, float32_output=float32_output
        )
    for seed in (15, 16):
        x.copy_(_inputs(6, n, k, seed=seed)[0])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            captured,
            skinny_fp16_linear_apply(x, w, bias, tile, float32_output=float32_output),
        )


# --- GPU tests: the DeepSeek-V4 router on FP16 logits --------------------------

ROUTER_EXPERTS, ROUTER_TOP_K, ROUTER_VOCAB = 256, 6, 512


def _route(logits, bias, input_ids=None, table=None):
    return fused_topk_bias(
        hidden_states=logits,  # only its row count is read
        gating_output=logits,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=bias,
        topk=ROUTER_TOP_K,
        renormalize=True,
        input_tokens=input_ids,
        hash_indices_table=table,
        routed_scaling_factor=1.5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [5, 6])
def test_router_gives_the_same_routing_for_fp16_logits(rows):
    """topk_softplus_sqrt on FP16 logits against their FP32 widening: torch.equal
    topk_ids and topk_weights (the kernel widens in registers; the argmax breaks
    ties by the lowest expert, the sum over the selected scores runs in k order)."""
    gen = torch.Generator(device="cuda").manual_seed(rows)
    bias = torch.randn(ROUTER_EXPERTS, generator=gen, device="cuda") * 0.1
    for _ in range(256):
        logits = (
            torch.randn(rows, ROUTER_EXPERTS, generator=gen, device="cuda") * 1.5
        ).half()
        weights16, ids16 = _route(logits, bias)
        weights32, ids32 = _route(logits.float(), bias)
        assert torch.equal(ids16, ids32)
        assert torch.equal(weights16, weights32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [5, 6])
def test_hash_router_selects_the_same_experts_for_fp16_logits(rows):
    """The hash layers take their ids from the table, so those never differ. The
    weights are summed across threads by the butterfly of the kernel, whose
    grouping depends on the input dtype: they may differ in the last bit, which is
    why DeepseekV4MoE keeps the FP32 copy for hash layers (widen_in_router False)."""
    gen = torch.Generator(device="cuda").manual_seed(100 + rows)
    table = torch.stack(
        [
            torch.randperm(ROUTER_EXPERTS, device="cuda")[:ROUTER_TOP_K]
            for _ in range(ROUTER_VOCAB)
        ]
    ).int()
    for _ in range(256):
        logits = (
            torch.randn(rows, ROUTER_EXPERTS, generator=gen, device="cuda") * 1.5
        ).half()
        input_ids = torch.randint(
            0, ROUTER_VOCAB, (rows,), generator=gen, device="cuda", dtype=torch.int32
        )
        weights16, ids16 = _route(logits, None, input_ids, table)
        weights32, ids32 = _route(logits.float(), None, input_ids, table)
        assert torch.equal(ids16, ids32)
        torch.testing.assert_close(weights16, weights32, rtol=4e-7, atol=0)
