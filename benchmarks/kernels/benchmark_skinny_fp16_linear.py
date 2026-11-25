# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cuBLAS against the skinny FP16 linear kernel on the decode layers of
DeepSeek-V4-Flash and GLM-5.3-Flash.

Every case runs inside a CUDA graph of back-to-back calls, so launch gaps do not
count. A shape cycles through as many distinct weight tensors as the model has
layers of that shape, and at least ``--rotation-mb`` of weights, so every call
streams its weight from DRAM instead of the 6 MB L2. The cuBLAS arm runs under
the worker's matmul policy (``set_high_precision_cuda_matmul_defaults``): no
reduced-precision FP16 reductions, the same numerics as in serving.

Reported per shape, row count and arm: microseconds per call (less is better),
weight bytes / time in GB/s (more is better), the largest absolute error against
an FP64 product of the same inputs and the number of elements more than one ULP
of the output dtype off it (fewer is better). ``--sweep`` times a grid of skinny
tiles instead and prints the fastest per shape (sum over the rows) and how far
the tile of the rule (``SKINNY_ARCH_RULES`` in skinny_fp16_linear.py) is behind
the fastest, to fit the rule per card. The default run measures the tile the rule
gives, also for a shape whose decision leaves it to cuBLAS (weight above the
limit), so that the limit can be checked against the numbers.

Run on one idle card (GPU-Orc hold first), once per card:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> \\
        python benchmarks/kernels/benchmark_skinny_fp16_linear.py --device cuda:0
Needs the Triton of kernel_config.triton_sm7x_tensor_cores on PYTHONPATH.
"""

import argparse
import itertools
import json
import statistics
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from triton.runtime.errors import OutOfResources

from vllm.model_executor.kernels.linear.skinny_fp16_linear import (
    SkinnyTile,
    skinny_fp16_linear,
    skinny_fp16_linear_device_traits,
    skinny_fp16_linear_layer_decision,
    skinny_fp16_linear_tile,
)
from vllm.utils.torch_utils import set_high_precision_cuda_matmul_defaults

PEAK_GBPS = {"V100": 900.0, "RTX 8000": 672.0}
MANTISSA_BITS = {torch.float16: 10, torch.float32: 23}


@dataclass(frozen=True)
class Shape:
    name: str
    n: int
    k: int
    out_dtype: torch.dtype  # what the model's call returns today
    bias: bool
    layers: int  # distinct weight tensors of this shape in the model
    source: str  # where the model calls it


# Production shapes (hidden 4096 for both models). The product of the compressor
# and indexer-compressor is FP32 out of cuBLAS (torch.mm(..., out_dtype=FP32)),
# the others are F.linear in FP16. Layer counts: DeepSeek-V4-Flash 43 layers,
# compress ratios 4 (21 layers) and 128 (20 layers) per config.json; GLM-5.3-Flash
# 45 layers + 1 MTP layer, 34 KDA layers, 11 sparse-attention layers + 1 MTP.
SHAPES = (
    Shape(
        "dsv4_router_gate",
        256,
        4096,
        torch.float16,
        False,
        43,
        "vllm/models/deepseek_v4/nvidia/model.py:443 GateLinear",
    ),
    Shape(
        "dsv4_compressor_c4",
        2 * 1024,
        4096,
        torch.float32,
        False,
        21,
        "vllm/models/deepseek_v4/compressor.py:297 fused_wkv_wgate",
    ),
    Shape(
        "dsv4_compressor_c128",
        2 * 512,
        4096,
        torch.float32,
        False,
        20,
        "vllm/models/deepseek_v4/compressor.py:297 fused_wkv_wgate",
    ),
    Shape(
        "dsv4_indexer_compressor",
        2 * 256,
        4096,
        torch.float32,
        False,
        21,
        "vllm/models/deepseek_v4/compressor.py:297 (indexer.compressor)",
    ),
    Shape(
        "dsv4_indexer_weights_proj",
        64,
        4096,
        torch.float16,
        False,
        21,
        "vllm/models/deepseek_v4/attention.py:950 weights_proj",
    ),
    Shape(
        "dsv4_dspark_markov_w2",
        129280,
        256,
        torch.float16,
        False,
        1,
        "vllm/models/deepseek_v4/nvidia/dspark.py:61 markov_w2 (draft head)",
    ),
    Shape(
        "glm_router_gate",
        288,
        4096,
        torch.float16,
        False,
        43,
        "vllm/models/glm5next/nvidia/model.py:272 GateLinear",
    ),
    Shape(
        "glm_kda_bfg_a",
        64 + 128 + 128,
        4096,
        torch.float16,
        False,
        34,
        "vllm/models/glm5next/nvidia/kda.py:424 in_proj_bfg_a",
    ),
    Shape(
        "glm_kda_f_b",
        8192,
        128,
        torch.float16,
        False,
        34,
        "vllm/models/glm5next/nvidia/kda.py:457 f_b_proj",
    ),
    Shape(
        "glm_kda_g_b",
        8192,
        128,
        torch.float16,
        False,
        34,
        "vllm/models/glm5next/nvidia/kda.py:507 g_b_proj",
    ),
    Shape(
        "glm_indexer_wk_weights_proj",
        128 + 32,
        4096,
        torch.float16,
        False,
        12,
        "vllm/models/glm5next/nvidia/attention.py:280 wk_weights_proj",
    ),
    Shape(
        "glm_indexer_wq_b",
        4096,
        1536,
        torch.float16,
        False,
        11,
        "vllm/models/glm5next/nvidia/attention.py:271 wq_b",
    ),
    Shape(
        "glm_indexer_kpool_gate",
        128,
        4096,
        torch.float16,
        False,
        12,
        "vllm/models/glm5next/nvidia/attention.py:410 index_kpool_compress_gate",
    ),
    Shape(
        "glm_mtp_eh_proj",
        4096,
        8192,
        torch.float16,
        False,
        1,
        "vllm/models/glm5next/nvidia/mtp.py:48 eh_proj",
    ),
)
SHAPES_BY_NAME = {shape.name: shape for shape in SHAPES}


def graph_time_us(launch, calls: int, repeats: int) -> float:
    """Median over ``repeats`` replays of one graph of ``calls`` launches,
    microseconds per launch. ``launch(index)`` issues call number ``index``."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for index in range(calls):  # compile and warm up outside the capture
            launch(index)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for index in range(calls):
            launch(index)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / calls)
    return statistics.median(times)


def rotation_copies(shape: Shape, rotation_mb: float) -> int:
    weight_bytes = shape.n * shape.k * 2
    return max(shape.layers, -(-int(rotation_mb * 1e6) // weight_bytes))


def make_weights(
    shape: Shape, copies: int, std: float, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weights = []
    for _ in range(copies):
        w = (torch.randn(shape.n, shape.k, device="cuda", generator=gen) * std).half()
        bias = (
            (torch.randn(shape.n, device="cuda", generator=gen) * std).half()
            if shape.bias
            else None
        )
        weights.append((w, bias))
    return weights


def cublas_call(shape: Shape, x, w, bias) -> torch.Tensor:
    """The call the model makes today for this shape."""
    if shape.out_dtype == torch.float32:
        return torch.mm(x, w.T, out_dtype=torch.float32)
    return F.linear(x, w, bias)


def ulp_errors(out: torch.Tensor, reference: torch.Tensor) -> tuple[float, int]:
    """(largest absolute error, elements more than 1 ULP of ``out``'s dtype
    away) against ``reference`` (FP64)."""
    error = (out.double() - reference).abs()
    exponent = torch.floor(torch.log2(reference.abs().clamp_min(2.0**-14)))
    ulp = torch.exp2(exponent - MANTISSA_BITS[out.dtype])
    return error.max().item(), int((error > ulp).sum().item())


def check_arms(shape, rows, tile, weights):
    """Largest error and ULP count of both arms on the first weight."""
    w, bias = weights[0]
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, shape.k, device="cuda", generator=gen).half()
    reference = x.double() @ w.double().T
    if bias is not None:
        reference += bias.double()
    out = torch.empty(rows, shape.n, dtype=shape.out_dtype, device="cuda")
    skinny_fp16_linear(x, w, out, tile, bias)
    torch.cuda.synchronize()
    return {
        "cublas": ulp_errors(cublas_call(shape, x, w, bias), reference),
        "skinny": ulp_errors(out, reference),
    }


def time_arms(shape, rows, tile, weights, args, arms=("cublas", "skinny")):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, shape.k, device="cuda", generator=gen).half()
    out = torch.empty(rows, shape.n, dtype=shape.out_dtype, device="cuda")
    calls = max(args.calls, len(weights))

    def cublas(index):
        w, bias = weights[index % len(weights)]
        cublas_call(shape, x, w, bias)

    def skinny(index):
        w, bias = weights[index % len(weights)]
        skinny_fp16_linear(x, w, out, tile, bias)

    launches = {"cublas": cublas, "skinny": skinny}
    return {arm: graph_time_us(launches[arm], calls, args.repeats) for arm in arms}


def decide(shape: Shape, weights, capability: int, sm_count: int):
    """(the tile of the rule, whether the layer decision admits the weight and
    why not)."""
    w, bias = weights[0]
    admitted = skinny_fp16_linear_layer_decision(
        w, bias, shape.out_dtype, capability=capability, sm_count=sm_count
    )
    rule = skinny_fp16_linear_tile(shape.n, shape.k, capability, sm_count)
    return rule.tile, admitted.tile is not None, admitted.reason


def run_shape(shape, args, traits, peak, records):
    copies = rotation_copies(shape, args.rotation_mb)
    weights = make_weights(shape, copies, args.weight_std, args.seed)
    weight_bytes = shape.n * shape.k * 2
    tile, admitted, reason = decide(shape, weights, *traits)
    print(
        f"\n{shape.name}: N={shape.n} K={shape.k} out={shape.out_dtype} "
        f"{copies} weights in rotation ({copies * weight_bytes / 1e6:.0f} MB), "
        f"weight {weight_bytes / 1e6:.2f} MB, peak {peak} GB/s  [{shape.source}]"
    )
    if tile is None:
        print(f"  the rule has no tile for this shape: {reason}")
        return
    if not admitted:
        print(f"  the layer decision leaves this shape to cuBLAS: {reason}")
    for rows in args.rows:
        times = time_arms(shape, rows, tile, weights, args)
        errors = check_arms(shape, rows, tile, weights)
        for arm in ("cublas", "skinny"):
            max_error, over_ulp = errors[arm]
            microseconds = times[arm]
            print(
                f"  M={rows} {arm:6s} {microseconds:8.2f} us "
                f"{weight_bytes / microseconds / 1000:7.1f} GB/s "
                f"max|err|={max_error:.3g} >1ulp={over_ulp}"
                + (f"  tile={tile}" if arm == "skinny" else "")
            )
            records.append(
                {
                    "shape": shape.name,
                    "rows": rows,
                    "arm": arm,
                    "us": microseconds,
                    "gbps": weight_bytes / microseconds / 1000,
                    "max_abs_error": max_error,
                    "elements_over_1ulp": over_ulp,
                    "tile": tile._asdict() if arm == "skinny" else None,
                    "admitted": admitted,
                }
            )


def tile_grid(shape: Shape, args):
    for block_n, block_k, warps, split in itertools.product(
        args.block_n, args.block_k, args.warps, args.split_k
    ):
        if (
            shape.k % (block_k * split)
            or shape.k // (block_k * split) > args.max_chunks
        ):
            continue
        yield SkinnyTile(block_n, block_k, warps, split)


def sweep_shape(shape, args, traits, peak, records):
    copies = rotation_copies(shape, args.rotation_mb)
    weights = make_weights(shape, copies, args.weight_std, args.seed)
    weight_bytes = shape.n * shape.k * 2
    print(f"\n{shape.name}: N={shape.n} K={shape.k} peak {peak} GB/s, rows {args.rows}")
    cublas_sum = sum(
        time_arms(shape, rows, None, weights, args, arms=("cublas",))["cublas"]
        for rows in args.rows
    )
    print(f"  cublas sum over rows: {cublas_sum:.2f} us")
    results = []
    for tile in tile_grid(shape, args):
        try:
            per_row = [
                time_arms(shape, rows, tile, weights, args, arms=("skinny",))["skinny"]
                for rows in args.rows
            ]
        except OutOfResources as error:
            # The grid includes tiles larger than a card's shared memory
            # (Turing: 64 KiB); they are not candidates there.
            print(f"  {tile} skipped: {error}")
            continue
        total = sum(per_row)
        results.append((total, tile, per_row))
        print(
            f"  {tile} sum {total:8.2f} us  " + " ".join(f"{t:6.2f}" for t in per_row)
        )
        records.append(
            {
                "shape": shape.name,
                "tile": tile._asdict(),
                "rows": list(args.rows),
                "us_per_row": per_row,
                "sum_us": total,
                "cublas_sum_us": cublas_sum,
            }
        )
    results.sort(key=lambda item: item[0])
    print(f"  fastest for {shape.name} (weight {weight_bytes / 1e6:.2f} MB):")
    for total, tile, _ in results[:3]:
        print(f"    {tile} sum {total:.2f} us ({cublas_sum / total:.2f}x cuBLAS)")
    rule_tile = skinny_fp16_linear_tile(shape.n, shape.k, *traits).tile
    if rule_tile is not None:
        rule_sum = sum(
            time_arms(shape, rows, rule_tile, weights, args, arms=("skinny",))["skinny"]
            for rows in args.rows
        )
        print(
            f"  rule tile {rule_tile} sum {rule_sum:.2f} us, "
            f"{(rule_sum / results[0][0] - 1) * 100:+.1f} % against the fastest, "
            f"{cublas_sum / rule_sum:.2f}x cuBLAS"
        )
        records.append(
            {
                "shape": shape.name,
                "rule_tile": rule_tile._asdict(),
                "rule_sum_us": rule_sum,
                "best_sum_us": results[0][0],
                "cublas_sum_us": cublas_sum,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shapes", nargs="+", choices=list(SHAPES_BY_NAME))
    parser.add_argument("--list", action="store_true", help="print the shapes, exit")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 5, 6, 7, 8])
    parser.add_argument("--calls", type=int, default=40, help="minimum per graph")
    parser.add_argument("--repeats", type=int, default=15, help="graph replays")
    parser.add_argument("--rotation-mb", type=float, default=64.0)
    parser.add_argument("--weight-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true", help="tile grid, skinny only")
    parser.add_argument("--block-n", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--block-k", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--warps", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--split-k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--max-chunks", type=int, default=8, help="sweep: unroll cap")
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args()

    selected = [SHAPES_BY_NAME[n] for n in args.shapes] if args.shapes else SHAPES
    if args.list:
        for shape in selected:
            print(
                f"{shape.name:30s} N={shape.n:5d} K={shape.k:5d} "
                f"out={shape.out_dtype} layers={shape.layers:3d}  {shape.source}"
            )
        return

    torch.cuda.set_device(args.device)
    set_high_precision_cuda_matmul_defaults()
    matmul = torch.backends.cuda.matmul
    device = torch.device(args.device)
    traits = skinny_fp16_linear_device_traits(device)
    name = torch.cuda.get_device_name(device)
    peak = next((v for key, v in PEAK_GBPS.items() if key in name), None)
    print(
        f"device {name} (capability {traits[0]}, {traits[1]} SMs), "
        f"peak {peak} GB/s; matmul policy: fp16 reduced-precision reduction "
        f"{matmul.allow_fp16_reduced_precision_reduction}, fp16 accumulation "
        f"{getattr(matmul, 'allow_fp16_accumulation', 'n/a')}"
    )
    records: list[dict] = []
    for shape in selected:
        if args.sweep:
            sweep_shape(shape, args, traits, peak, records)
        else:
            run_shape(shape, args, traits, peak, records)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": name,
                    "capability": traits[0],
                    "sm_count": traits[1],
                    "shapes": [
                        {**asdict(s), "out_dtype": str(s.out_dtype)} for s in selected
                    ],
                    "results": records,
                },
                handle,
                indent=1,
            )
        print(f"\nresults written to {args.json}")


if __name__ == "__main__":
    main()
