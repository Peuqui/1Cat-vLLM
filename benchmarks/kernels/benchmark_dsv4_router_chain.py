# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4 router chain of one decode step, step by step and fused.

Today's chain per MoE layer (rows = the verify batch, 5 or 6):

    gate     F.linear(hidden FP16, gate.weight FP16 [256, 4096])   -> FP16 logits
    to_fp32  logits.to(torch.float32)               (GateLinear out_dtype FP32)
    topk     fused_topk_bias -> torch.ops._moe_C.topk_softplus_sqrt: sqrtsoftplus,
             e_score_correction_bias FP32, top-6, renormalize, scaling 1.5
    group    group_routing(topk_ids) (vllm/.../fused_moe/sm70/routing.py)

Variants, each a whole chain over the same layers:
    A  today's chain.
    B  A without ``to_fp32``: the FP16 logits go straight into the top-k kernel
       (it reads FP16 and converts to FP32 in registers, csrc/moe/
       topk_softplus_sqrt_kernels.cu:209-229; FP16 -> FP32 is exact).
    C  A with the gate from the skinny kernel (FP16 out like F.linear).
    C2 B with the gate from the skinny kernel.
    D  skinny gate with FP32 out (no FP16 rounding of the logits) into the FP32
       top-k. Not bit-equal to A by construction; reported only to show how
       often the routing differs.
    P  what the model runs with ``kernel_config.skinny_fp16_linear`` (final
       variant): the gate through the opaque op ``torch.ops.vllm.skinny_fp16_linear``
       with the layer's rule tile, its FP16 logits straight into the top-k
       (``GateLinear(widen_in_router=True)``; the three hash layers keep the FP32
       copy and are not measured here).

Every step is also timed alone in a CUDA graph over all layers' tensors in
rotation (distinct gate weights and bias vectors per layer, so the gate streams
its weight from DRAM). Equality: for every layer and ``--trials`` hidden states,
B must give ``torch.equal`` topk_ids and topk_weights to A (the ids and weights
follow from the kernel source: per-element math on the exact FP32 value, argmax
with lowest-index tie-break, k-sequential sum), and C/C2/D are compared to A.

Not covered: the three hash layers (tid2eid, no bias) take the USE_HASH branch of
the top-k kernel; they are left out. The gate weights are random (``--weight-std``,
``--bias-std``) unless ``--checkpoint`` points at the DeepSeek-V4 checkpoint, whose
``layers.N.ffn.gate.{weight,bias}`` (BF16 -> FP16 like ``--dtype half``) are used.

Run on one idle card (GPU-Orc hold first), once per card:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> \\
        python benchmarks/kernels/benchmark_dsv4_router_chain.py --device cuda:0
Needs the Triton of kernel_config.triton_sm7x_tensor_cores on PYTHONPATH.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark_skinny_fp16_linear import graph_time_us

from vllm.model_executor.kernels.linear.skinny_fp16_linear import (
    skinny_fp16_linear,
    skinny_fp16_linear_apply,
    skinny_fp16_linear_decision,
    skinny_fp16_linear_device_traits,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.sm70.routing import group_routing
from vllm.utils.torch_utils import set_high_precision_cuda_matmul_defaults

HIDDEN, EXPERTS, TOP_K = 4096, 256, 6
ROUTED_SCALING_FACTOR = 1.5
SCORING_FUNC = "sqrtsoftplus"
LAYERS = 43  # DeepSeek-V4-Flash; the three hash layers have no bias (see above)
HASH_LAYERS = 3


def random_gates(layers: int, weight_std: float, bias_std: float, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return [
        (
            (
                torch.randn(EXPERTS, HIDDEN, device="cuda", generator=gen) * weight_std
            ).half(),
            torch.randn(EXPERTS, device="cuda", generator=gen) * bias_std,
        )
        for _ in range(layers)
    ]


def checkpoint_gates(path: str):
    from safetensors import safe_open

    root = Path(path)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    gates = []
    for layer in range(LAYERS):
        weight_name = f"layers.{layer}.ffn.gate.weight"
        bias_name = f"layers.{layer}.ffn.gate.bias"
        if bias_name not in index:  # hash layer: tid2eid instead of a bias
            continue
        with safe_open(root / index[weight_name], framework="pt") as handle:
            weight = handle.get_tensor(weight_name)
        with safe_open(root / index[bias_name], framework="pt") as handle:
            bias = handle.get_tensor(bias_name)
        gates.append((weight.to("cuda", torch.float16), bias.to("cuda", torch.float32)))
    return gates


def route(logits: torch.Tensor, bias: torch.Tensor):
    """fused_topk_bias as the DeepSeek-V4 MoE calls it (no hash table)."""
    return fused_topk_bias(
        hidden_states=logits,  # only its row count is read
        gating_output=logits,
        scoring_func=SCORING_FUNC,
        e_score_correction_bias=bias,
        topk=TOP_K,
        renormalize=True,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )


class Chains:
    """Steps and variants for one row count, over all layers' gates."""

    def __init__(self, gates, rows: int, traits, seed: int):
        self.gates, self.rows = gates, rows
        gen = torch.Generator(device="cuda").manual_seed(seed + rows)
        self.hidden = torch.randn(rows, HIDDEN, device="cuda", generator=gen).half()
        weight = gates[0][0]
        decision = skinny_fp16_linear_decision(
            self.hidden,
            weight,
            None,
            torch.float16,
            capability=traits[0],
            sm_count=traits[1],
        )
        if decision.tile is None:
            raise ValueError(
                f"skinny kernel does not cover the gate: {decision.reason}"
            )
        self.tile = decision.tile
        # Per-layer inputs of the steps that run alone.
        self.logits16 = [F.linear(self.hidden, w) for w, _ in gates]
        self.logits32 = [logits.float() for logits in self.logits16]
        self.ids = [route(logits, b)[1] for logits, (_, b) in zip(self.logits32, gates)]

    def skinny_gate(
        self, layer: int, hidden: torch.Tensor, out_dtype=torch.float16
    ) -> torch.Tensor:
        out = torch.empty(self.rows, EXPERTS, dtype=out_dtype, device="cuda")
        skinny_fp16_linear(hidden, self.gates[layer][0], out, self.tile)
        return out

    # steps (the index is the layer, cycled)
    def step_gate_cublas(self, i):
        F.linear(self.hidden, self.gates[i % len(self.gates)][0])

    def step_gate_skinny(self, i):
        self.skinny_gate(i % len(self.gates), self.hidden)

    def step_gate_op(self, i):
        skinny_fp16_linear_apply(
            self.hidden, self.gates[i % len(self.gates)][0], None, self.tile
        )

    def step_to_fp32(self, i):
        self.logits16[i % len(self.gates)].to(torch.float32)

    def step_topk_fp32(self, i):
        layer = i % len(self.gates)
        route(self.logits32[layer], self.gates[layer][1])

    def step_topk_fp16(self, i):
        layer = i % len(self.gates)
        route(self.logits16[layer], self.gates[layer][1])

    def step_group(self, i):
        group_routing(self.ids[i % len(self.gates)])

    # whole chains
    def launcher(self, variant: str):
        return lambda i: self.chain(variant, i % len(self.gates), self.hidden)

    def chain(self, variant: str, layer: int, hidden: torch.Tensor):
        weight, bias = self.gates[layer]
        if variant in ("A", "B"):
            logits = F.linear(hidden, weight)
        elif variant == "D":
            logits = self.skinny_gate(layer, hidden, torch.float32)
        elif variant == "P":
            logits = skinny_fp16_linear_apply(hidden, weight, None, self.tile)
        else:
            logits = self.skinny_gate(layer, hidden)
        if variant in ("A", "C"):
            logits = logits.to(torch.float32)
        topk_weights, topk_ids = route(logits, bias)
        group_routing(topk_ids)
        return topk_weights, topk_ids


STEPS = (
    ("gate (cuBLAS F.linear)", "step_gate_cublas"),
    ("gate (skinny)", "step_gate_skinny"),
    ("gate (skinny, through the op)", "step_gate_op"),
    ("to_fp32", "step_to_fp32"),
    ("topk, FP32 logits", "step_topk_fp32"),
    ("topk, FP16 logits", "step_topk_fp16"),
    ("group_routing", "step_group"),
)
VARIANTS = (
    ("A", "today: cuBLAS gate -> to_fp32 -> topk -> group"),
    ("B", "A without to_fp32 (FP16 logits into topk)"),
    ("C", "skinny gate -> to_fp32 -> topk -> group"),
    ("C2", "skinny gate -> topk on FP16 logits -> group"),
    ("D", "skinny gate FP32 out -> topk -> group (not bit-equal to A)"),
    ("P", "final: gate through the op -> topk on FP16 logits -> group"),
)


def compare(chains: Chains, trials: int, seed: int) -> None:
    """torch.equal of every variant's routing against A, per layer and hidden."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 1000 + chains.rows)
    mismatches = {name: {"ids": 0, "weights": 0, "max_dw": 0.0} for name, _ in VARIANTS}
    total = 0
    for layer in range(len(chains.gates)):
        for _ in range(trials):
            hidden = torch.randn(
                chains.rows, HIDDEN, device="cuda", generator=gen
            ).half()
            reference = chains.chain("A", layer, hidden)
            total += 1
            for name, _ in VARIANTS[1:]:
                weights, ids = chains.chain(name, layer, hidden)
                row = mismatches[name]
                ids_equal = torch.equal(ids, reference[1])
                row["ids"] += not ids_equal
                row["weights"] += not torch.equal(weights, reference[0])
                if ids_equal:
                    row["max_dw"] = max(
                        row["max_dw"], (weights - reference[0]).abs().max().item()
                    )
    print(f"  equality against A over {total} (layer, hidden) pairs:")
    for name, description in VARIANTS[1:]:
        row = mismatches[name]
        print(
            f"    {name:3s} topk_ids differ {row['ids']:4d}, topk_weights not "
            f"torch.equal {row['weights']:4d}, max|dw| where ids equal "
            f"{row['max_dw']:.3g}   [{description}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--calls", type=int, default=LAYERS, help="per graph")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--trials", type=int, default=8, help="hidden states/layer")
    parser.add_argument("--weight-std", type=float, default=0.02)
    parser.add_argument("--bias-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", help="DeepSeek-V4 checkpoint directory")
    parser.add_argument("--json", help="write the timings to this file")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    set_high_precision_cuda_matmul_defaults()
    device = torch.device(args.device)
    traits = skinny_fp16_linear_device_traits(device)
    gates = (
        checkpoint_gates(args.checkpoint)
        if args.checkpoint
        else random_gates(LAYERS, args.weight_std, args.bias_std, args.seed)
    )
    gate_mb = gates[0][0].numel() * 2 * len(gates) / 1e6
    print(
        f"device {torch.cuda.get_device_name(device)} (capability {traits[0]}, "
        f"{traits[1]} SMs), {len(gates)} gate layers "
        f"({'checkpoint' if args.checkpoint else 'random weights'}; hash layers "
        f"not included), gate weights {gate_mb:.0f} MB"
    )
    records: list[dict] = []
    for rows in args.rows:
        chains = Chains(gates, rows, traits, args.seed)
        calls = max(args.calls, len(gates))
        print(f"\nM={rows}, skinny tile {chains.tile}")
        print("  steps alone (us per call, less is better):")
        for label, method in STEPS:
            microseconds = graph_time_us(getattr(chains, method), calls, args.repeats)
            print(f"    {label:24s} {microseconds:8.2f}")
            records.append({"rows": rows, "step": label, "us": microseconds})
        print("  whole chains (us per layer, less is better):")
        for name, description in VARIANTS:
            microseconds = graph_time_us(
                chains.launcher(name),
                calls,
                args.repeats,
            )
            print(f"    {name:3s} {microseconds:8.2f}   {description}")
            records.append({"rows": rows, "variant": name, "us": microseconds})
        compare(chains, args.trials, args.seed)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {"device": str(device), "capability": traits[0], "results": records},
                handle,
                indent=1,
            )
        print(f"\nresults written to {args.json}")


if __name__ == "__main__":
    main()
