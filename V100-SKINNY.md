# v100-skinny on 1Cat-vLLM 1.5.0

This branch is 1Cat-vLLM `v1.5.0` plus the v100-skinny changes as a source tree:
the Volta/Turing (SM70/SM75) serving work for a heterogeneous box with
2x Quadro RTX 8000 and 3x Tesla V100 (NVFP4 skinny GEMM/MoE kernels, the SM75
GDN and attention paths, pipeline-parallel speculative decoding, DeepSeek-V4
DSpark under PP, memory-profiling and calibration fixes). It is generated from
the overlay in https://github.com/Peuqui/v100-skinny (`fork_patches_150/`,
`fork_patches/flash_linear_attention/`, `kernels/skinny_kernels.cu`) and is
meant for reading, diffing and cherry-picking. Upstream-ready slices are
proposed separately as pull requests against 1CatAI/1Cat-vLLM main.

Base: tag `v1.5.0` (d8f42b3). Python only; the compiled 1Cat wheel for
`v1.5.0` serves as the binary base.

## Using it

Overlay (no CUDA build, what we run): install the 1Cat-vLLM 1.5.0 wheel into
a venv, clone Peuqui/v100-skinny and run
`ENV_PREFIX=<venv> scripts/deploy-fork-patches-150.sh`. The script copies
the same files this branch contains over the installed package and keeps a
`.pre_deploy` copy of every replaced file.

Source checkout: build this branch like 1Cat-vLLM itself, or point
`PYTHONPATH` at it on top of an installed `v1.5.0` wheel (all changes are
Python and one CUDA source that is JIT-compiled at runtime).

One patch cannot live in this tree: `fork_patches_150/tilelang_target.py`
replaces `tilelang/utils/target.py` inside the tilelang package and stays
overlay-only.

## Switches

Everything is opt-in through environment variables (`VLLM_SKINNY_NVFP4=1`,
`VLLM_SKINNY_QPN=1`, `VLLM_SKINNY_QPN2=1`, `VLLM_SKINNY_DROP_CT=1`,
`VLLM_SKINNY_DENSE_PREFILL=1`, `VLLM_SM70_QUANT_BACKEND=auto`, ...). The
serving recipes, measurements and the handover notes live in the
v100-skinny repository (`scripts/serve-*.sh`, `TURING-COEXISTENCE-HANDOVER.md`,
`MERGE-PROJECT-HANDOVER.md`).

Hardware-specific findings in the code (device-0 capability gates, the gloo
transport for spec-decode state over five PP stages, SM75 dequant paths) are
documented at the change with a `Fork fix (v100-skinny)` comment.
