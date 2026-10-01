# Running this fork on Volta and Turing (V100, RTX 8000)

This branch is 1Cat-vLLM main plus the open pull requests of this fork. It
runs DeepSeek-V4-Flash, Qwen3.8-Flash-Next and Qwen3.8-27B on Tesla V100
(sm_70), Quadro RTX 8000 and other Turing cards (sm_75), and mixed rigs of
both. This page covers how to build it, which extra library Turing needs,
and how to start the models with the switches that make them fast.

Reference rig: 3x Tesla V100-PCIE-32GB + 2x Quadro RTX 8000 (48 GB), every
card on PCIe Gen3 x4, no P2P, NVIDIA driver 580, Ubuntu 24.04, Python 3.12.

Verified on 2026-10-01 with a fresh clone of this branch (commit 20618af1),
a fresh venv and empty compile caches, following this page step by step:
Qwen3.8-27B booted on two RTX 8000 and produced output identical to the
production build of the same commit (greedy and eight longer prompts); all
13 compiled extension modules match the production build.

## 0. Get the code

```bash
git clone --branch fork-union https://github.com/Peuqui/1Cat-vLLM.git
cd 1Cat-vLLM
```

All later commands run in this directory unless they say otherwise.

## 1. Toolchain

- CUDA toolkit 12.8 (`nvcc` 12.8). Set `CUDA_HOME` to it.
- Python 3.12 and a fresh venv.
- torch 2.10.0 with CUDA 12.8. It is the last wheel series that still
  contains sm_70.

```bash
python3.12 -m venv ~/venvs/1cat
source ~/venvs/1cat/bin/activate
pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/build/cuda.txt
pip install -r requirements/cuda.txt
# CCCL headers matching CUDA 12 (see step 2). The cu13 CCCL that some of the
# requirements pull in does not fit nvcc 12.8.
pip install nvidia-cuda-cccl-cu12==12.9.27
```

After every (re)installation of torch, apply the backports in
`tools/torch_patches` (see its README). Delete `torch_aot_compile/` under the
vLLM cache root once afterwards:

```bash
tools/torch_patches/apply.sh "$(which python)"
```

## 2. Build vLLM for sm_70

Build for 7.0 only, even when Turing cards are present. The sm_70 kernels
run on Turing through binary compatibility. A combined `7.0;7.5` build drops
the SM70 Marlin and Marlin-MoE kernels, so the V100s lose them.

A CUDA 12.8 toolkit installed as a plain `nvcc` package lacks the CCCL
headers. The compiler then picks up the distribution's old libcu++ from
`/usr/include` and fails with `namespace "cuda::std" has no member
"isfinite"`. Point `CPATH` at the `nvidia-cuda-cccl-cu12` wheel installed in
step 1:

```bash
export CUDA_HOME=/path/to/cuda-12.8
export CPATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=4            # each nvcc job needs several GB of RAM
unset VLLM_FLASH_ATTN_SRC_DIR
pip install -e . --no-build-isolation
```

A full build takes about an hour with `MAX_JOBS=3`. Besides `vllm/_C` it
builds the Volta FlashAttention (`vllm/vllm_flash_attn/_vllm_fa2_C`), the
FlashAttention-V100 extension and the SM70 GDN kernels for `flash_qla`.

## 3. Turing only: FlashAttention-2 for sm_75

The sm_70 build above contains no FlashAttention for Turing. Without it, the
Qwen models on Turing fall back to the slower TRITON_ATTN backend.
DeepSeek-V4 does not need it.

Build the library separately with the arguments from
`cmake/external_projects/vllm_flash_attn_sm75.cmake`. Then place it next to
the Volta one; `flash_attn_interface.load_fa2_library` picks the file per
device.

```bash
git clone --recurse-submodules https://github.com/Peuqui/flash-attention.git fa-sm75
cd fa-sm75 && git checkout 43b9d29c9aa8e18d9351e7c643dc78ef7e7979fc  # tag sm75-1cat-2026-09-13
git submodule update --init --recursive csrc/cutlass
cmake -S . -B build -G Ninja \
    -DPython_EXECUTABLE="$(which python)" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDA_ARCHS=7.5 \
    -DFA2_ENABLED=ON -DFA3_ENABLED=OFF -DVLLM_FA2_OUTPUT_NAME=_vllm_fa2_C_sm75
cmake --build build --target _vllm_fa2_C -j "$MAX_JOBS"
cp build/_vllm_fa2_C_sm75.abi3.so /path/to/this/repo/vllm/vllm_flash_attn/
```

## 4. Checkpoints

| model | Hugging Face repository |
|---|---|
| DeepSeek-V4-Flash with the DSpark drafter | `deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| Qwen3.8-Flash-Next | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| Qwen3.8-27B | `RadixArk/Qwen3.8-27B-NVFP4` |

## 5. Starting the models

Common environment for all of them:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # stage order by bus, not by speed
export NCCL_P2P_DISABLE=1             # no P2P between these cards
export VLLM_NO_USAGE_STATS=1
```

`CUDA_VISIBLE_DEVICES` and `VLLM_PP_LAYER_PARTITION` below are for the
reference rig: device 0 and 2 are the RTX 8000s, 1, 3 and 4 the V100s.
Adapt them to your cards. Each pipeline stage needs room for its layers and
its share of the KV cache.

### DeepSeek-V4-Flash + DSpark, PP5

```bash
CUDA_VISIBLE_DEVICES=0,1,4,3,2 VLLM_PP_LAYER_PARTITION=10,8,8,8,9 \
VLLM_SM70_QUANT_BACKEND=marlin VLLM_SM70_NVFP4_TURBOMIND=0 \
VLLM_SM70_FP8_BLOCK_QPN8=1 \
VLLM_SM70_DSV4_SPARSE_MLA_BMM=1 VLLM_SM70_DSV4_SPARSE_MLA_BMM_PREFILL=1 \
VLLM_SM70_INDEXER_DECODE_CUBLAS=1 VLLM_SM70_INDEXER_PREFILL_TILE_MB=64 \
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/DeepSeek-V4-Flash-DSpark --trust-remote-code \
    --dtype half --kv-cache-dtype fp8 --disable-custom-all-reduce \
    --tensor-parallel-size 1 --pipeline-parallel-size 5 \
    --gpu-memory-utilization 0.95 --max-model-len 307200 \
    --max-num-seqs 1 --max-num-batched-tokens 512 \
    --moe-backend sm70_skinny \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
    --compilation-config '{"cudagraph_capture_sizes":[6],"max_cudagraph_capture_size":6}' \
    --enable-prefix-caching --safetensors-load-strategy direct \
    --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4
```

### Qwen3.8-Flash-Next, PP4 (also TP2 x PP2)

```bash
CUDA_VISIBLE_DEVICES=0,2,1,3 VLLM_PP_LAYER_PARTITION=12,12,12,12 \
VLLM_SM70_QUANT_BACKEND=auto VLLM_SM70_NVFP4_TURBOMIND=1 \
VLLM_SM70_NVFP4_MOE_GROUPED_MAX_TOKENS=2048 VLLM_SM70_NVFP4_MOE_QPN_CFG=16,1,10,1 \
NCCL_BUFFSIZE=1048576 \
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3.8-Flash-Next-NVFP4 --trust-remote-code \
    --dtype float16 --disable-custom-all-reduce \
    --tensor-parallel-size 1 --pipeline-parallel-size 4 \
    --gpu-memory-utilization 0.95 --block-size 16 --max-model-len 262144 \
    --max-num-seqs 4 --max-num-batched-tokens 2048 \
    --moe-backend sm70_skinny --async-scheduling \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"greedy"}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,5,8]}' \
    --enable-prefix-caching --safetensors-load-strategy direct \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

For TP2 x PP2, use `--tensor-parallel-size 2 --pipeline-parallel-size 2` and
`VLLM_PP_LAYER_PARTITION=24,24`. To spill the per-layer embedding table to
disk instead of host RAM, add `VLLM_QWEN4EXP_PLE_HOST_GIB=0
VLLM_QWEN4EXP_PLE_DISK=1 VLLM_PLE_DISK_RELEASE_PAGES=1`.

### Qwen3.8-27B, TP2

```bash
CUDA_VISIBLE_DEVICES=0,2 \
VLLM_SM70_QUANT_BACKEND=auto VLLM_SM70_NVFP4_TURBOMIND=1 \
VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1 NCCL_BUFFSIZE=1048576 \
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3.8-27B-NVFP4 --trust-remote-code \
    --dtype float16 --disable-custom-all-reduce \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.98 --block-size 16 \
    --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 2048 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy","use_local_argmax_reduction":true,"attention_backend":"FLASH_ATTN"}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' \
    --enable-prefix-caching --safetensors-load-strategy direct \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

## 6. What the switches do

| switch | effect |
|---|---|
| `--moe-backend sm70_skinny` | MoE through the skinny QPN kernels for NVFP4 and MXFP4 on Volta and Turing (#742) |
| `VLLM_SM70_FP8_BLOCK_QPN8=1` | all [128, 128] block-FP8 linears through the native QPN8 operators instead of TurboMind/Marlin (#750) |
| `VLLM_SM70_DSV4_SPARSE_MLA_BMM=1`, `..._BMM_PREFILL=1` | DeepSeek-V4 sparse MLA as gather + batched matmul (#716) |
| `VLLM_SM70_INDEXER_DECODE_CUBLAS=1` | DeepSeek-V4 indexer decode through cuBLAS, also under speculative decoding (#714) |
| `VLLM_SM70_QUANT_BACKEND` | `marlin`, `turbomind` or `auto` for the SM70 quantized linears |
| `--safetensors-load-strategy direct` | O_DIRECT loading, no page cache build-up (#740) |

DeepSeek-V4 on Turing and in mixed rigs needs #747 (SM70 route on Turing) and
#745 (software FP8 below sm_89). Both are part of this branch and need no
switch.

## 7. Notes

- The first request after a cold start is slow, because of graph capture
  and kernel autotuning. The first long prompt is not representative.
- With `VLLM_SM70_QUANT_BACKEND=auto` and TurboMind FP8 on Volta, every new
  prompt length makes TurboMind tune its FP8 GEMMs once, which takes seconds.
  For DeepSeek-V4, `VLLM_SM70_FP8_BLOCK_QPN8=1` avoids that.
- Keep `--dtype half`/`float16`: neither Volta nor Turing has BF16 tensor
  cores.
