# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM70 QPN8 kernel for BLOCK-scaled FP8 checkpoints (fork addition).

Serves blockwise-FP8 linears (weight_block_size, typically [128, 128]:
DeepSeek-class attention, HF-fp8 Qwen exports) on Volta through the skinny
QPN8 mma.m8n8k4 path. Without this, SM70 falls through the block-kernel
priority list to MarlinFP8ScaledMMLinearKernel (weight-only, 14.8 tok/s on
the 27B); the QPN8-blk kernel costs only 2-4% over the per-tile QPN8 path
(scale raster lookup at the decode point, see kernels/skinny_kernels.cu).

Weight-only: activations stay fp16 (apply_input_quant=False), the fp32
accumulate happens in the mma fragments. Prepack, unpack and the skinny
extension are shared with the per-tensor QPN8 path in
vllm/model_executor/layers/quantization/modelopt.py (lazy imports: modelopt
imports this package's __init__, so a top-level import would be circular).
"""

import os as _os

import torch
from torch.library import custom_op as _custom_op

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)

logger = init_logger(__name__)

_SM70_QPN8_BLK = _os.environ.get("VLLM_SM70_QPN8_BLK", "1") == "1"
# Same encoding as the per-tensor path: key splitk*10+nacc, +2 on nacc
# selects the fast decoder. split16/fast is the measured frontier on the
# production shapes; stash() falls back when K/16 does not divide.
_SM70_QPN8_BLK_CFG = tuple(
    int(v) for v in _os.environ.get("VLLM_SM70_QPN8_BLK_CFG", "16,3").split(","))
# Internal M-band frontier (benchmarks/fp8_blk_backend_bench.py sweep,
# DeepSeek attention shapes): native <=8, MT2 <=16, WMMA tiles up to this
# bound, transient dequant + cuBLAS hgemm above it. The wmma/dequant curves
# cross between M=256 and M=512 on every shape with <20% between them, so
# the measured-winner boundary 256 is used.
_SM70_QPN8_BLK_WMMA_MAX = int(
    _os.environ.get("VLLM_SM70_QPN8_BLK_WMMA_MAX", "256"))

def qpn8_blk_enabled() -> bool:
    """Gate shared with Fp8Config.get_min_capability (fork fp8.py patch)."""
    return _SM70_QPN8_BLK


_blk_verified_shapes: set = set()
_blk_census_seen: set = set()


@_custom_op("sm70_fp8::qpn8_blk_linear", mutates_args=())
def _qpn8_blk_linear(x: torch.Tensor, codes: torch.Tensor,
                     bscale: torch.Tensor, n: int, k: int, bn: int, bk: int,
                     splitk: int, nacc: int) -> torch.Tensor:
    from vllm.model_executor.kernels.linear.nvfp4.marlin import _get_skinny_ext
    ext = _get_skinny_ext()
    m = x.shape[0]
    use_wmma = (k % 128 == 0) and m <= _SM70_QPN8_BLK_WMMA_MAX
    _ck = (int(k), int(n), int(m))
    if _ck not in _blk_census_seen:
        _blk_census_seen.add(_ck)
        _rt = ("qpn8-blk" if m <= 8 else "qpn8-blk-mt2" if m <= 16
               else "qpn8-blk-wmma" if use_wmma else "qpn8-blk-dequant")
        logger.info("QPN8_BLK_CENSUS_RUN K=%d N=%d M=%d route=%s split=%d "
                    "nacc=%d", int(k), int(n), int(m), _rt, int(splitk),
                    int(nacc))
    if m <= 8:
        return ext.gemm_qpn8_blk(x, codes, bscale, n, bn, bk, splitk, nacc)
    if m <= 16:
        # MT2 caps SPLITK at 16 (shared-memory staging, see the kernel).
        msp = min(int(splitk), 16)
        return ext.gemm_qpn8_blk_mt2(x, codes, bscale, n, bn, bk, msp, nacc)
    if use_wmma:
        # mid band: tensor-core tiles straight from the packed layout
        return ext.gemm_qpn8_blk_wmma(x, codes, bscale, n, bn, bk)
    # large band: fast transient dequant + cuBLAS hgemm (never persisted)
    wf = ext.qpn8_blk_dequant(codes, bscale, n, k, bn, bk)
    return torch.nn.functional.linear(x, wf)


@_qpn8_blk_linear.register_fake
def _qpn8_blk_linear_fake(x, codes, bscale, n, k, bn, bk, splitk, nacc):
    return x.new_empty((x.shape[0], n))


class QPN8Fp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """Block-scaled FP8 on SM70 via the skinny QPN8 codec."""

    # fp16 activations go straight into the mma; no input quantization.
    apply_input_quant = False

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not _SM70_QPN8_BLK:
            return False, "disabled via VLLM_SM70_QPN8_BLK=0"
        if not torch.cuda.is_available():
            return False, "CUDA unavailable"
        # The LOCAL worker device decides, never device 0 of the visibility
        # list -- on the heterogeneous grid the stages differ (the Session-4
        # lesson from the SM70 baseline in config/vllm.py).
        # sm70 runs the QPN8 codec; sm75 (no fp8 kernels either, and the
        # QPN8 cubin is Volta-only) dequantizes to fp16 at load and runs
        # cuBLAS -- the RTX stages of the heterogeneous grid have the VRAM
        # headroom for that, and every PP stage carries block-FP8 attention.
        cap = torch.cuda.get_device_capability(torch.cuda.current_device())
        if cap not in ((7, 0), (7, 5)):
            return False, f"sm{cap[0]}{cap[1]} is not sm70/sm75"
        return True, None

    @classmethod
    def can_implement(cls, config):
        if config.input_dtype != torch.float16:
            return False, "QPN8-blk needs fp16 activations"
        n, k = config.weight_shape
        if n % 32 or k % 64:
            return False, f"QPN8 geometry needs N%32==0, K%64==0 (got {n},{k})"
        bn, bk = config.weight_quant_key.scale.group_shape
        if bn % 32 or bk % 16:
            return False, f"block size [{bn},{bk}] not tile/group aligned"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        from vllm.model_executor.layers.quantization.modelopt import (
            _sm70_qpn8_prepack,
            _sm70_qpn8_unpack,
        )
        params = self._get_layer_params(layer)
        w8 = params.weight.data
        n, k = w8.shape
        bn, bk = self.weight_group_shape
        scale = (params.weight_scale
                 if params.weight_scale_inv is None else params.weight_scale_inv)
        bscale = scale.data.detach().float().contiguous()
        if bscale.dim() != 2 or bscale.shape != ((n + bn - 1) // bn,
                                                (k + bk - 1) // bk):
            raise ValueError(
                f"QPN8-blk: scale raster {tuple(bscale.shape)} does not match "
                f"weights {n}x{k} at block [{bn},{bk}]")

        if getattr(layer, "is_bmm", False):
            # is_bmm layers (DeepSeek wo_a): the model consumes layer.weight
            # directly in a grouped einsum (the fp8_einsum path needs
            # DeepGEMM); fp16-dequant at load serves the reference einsum
            # on BOTH archs. Plain linears no longer dequant on sm75:
            # the packed QPN8 route beats fp16 cuBLAS ~2x at decode M<=8
            # on the RTX 8000 (m8n8k4 is memory-bound at these M, the
            # Turing-mma concern does not apply; see
            # benchmarks/fp8_blk_sm75_decode_bench.py, 2026-09-03).
            sc = (bscale.repeat_interleave(bn, 0)[:n]
                  .repeat_interleave(bk, 1)[:, :k])
            w16 = (w8.view(torch.float8_e4m3fn).to(torch.float32) * sc).half()
            layer.weight = torch.nn.Parameter(w16, requires_grad=False)
            layer._qpn8_dequant16 = True
            logger.info("QPN8_BLK_CENSUS_LOAD layer=%s K=%d N=%d "
                        "route=is_bmm-fp16-dequant",
                        getattr(layer, "prefix", "?"), k, n)
            return

        raw = w8.view(torch.uint8) if w8.dtype != torch.uint8 else w8
        packed = _sm70_qpn8_prepack(raw)
        if (n, k) not in _blk_verified_shapes:
            # Invert and assert byte identity: the original weight is freed
            # below, so the packed buffer becomes the only copy.
            assert packed.numel() == n * k, "qpn8-blk packed size"
            _rt = _sm70_qpn8_unpack(packed, n, k)
            assert torch.equal(_rt, raw), (
                "QPN8-blk prepack is not invertible for n=%d k=%d" % (n, k))
            _blk_verified_shapes.add((n, k))
            logger.info("QPN8_BLK prepack INVERTED and byte-identical: %s "
                        "n=%d k=%d", getattr(layer, "prefix", "?"), n, k)

        layer._qpn8_codes = packed
        layer._qpn8_bscale = bscale
        layer._qpn8_shape = (n, k)
        layer._qpn8_geom = (int(bn), int(bk))
        splitk, nacc = _SM70_QPN8_BLK_CFG
        if (k // 16) % splitk:
            splitk = 8 if (k // 16) % 8 == 0 else 4
        layer._qpn8_cfg = (splitk, nacc)
        logger.info("QPN8_BLK_CENSUS_LOAD layer=%s K=%d N=%d block=[%d,%d] "
                    "split=%d nacc=%d", getattr(layer, "prefix", "?"), k, n,
                    bn, bk, splitk, nacc)
        # The packed buffer is now the only resident copy.
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=torch.uint8, device=packed.device),
            requires_grad=False)

    def apply_block_scaled_mm(self, A, B, As, Bs):
        # Satisfies the ABC; apply_weights below bypasses the base-class
        # A/As machinery entirely (weight-only path, fp16 activations).
        raise RuntimeError("unreachable: QPN8-blk overrides apply_weights")

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: torch.Tensor | None = None,
                      **kwargs) -> torch.Tensor:
        if getattr(layer, "_qpn8_dequant16", False):
            return torch.nn.functional.linear(x, layer.weight, bias)
        n, k = layer._qpn8_shape
        bn, bk = layer._qpn8_geom
        splitk, nacc = layer._qpn8_cfg
        xc = x.reshape(-1, k).contiguous()
        y = torch.ops.sm70_fp8.qpn8_blk_linear(
            xc, layer._qpn8_codes, layer._qpn8_bscale, n, k, bn, bk,
            splitk, nacc)
        if bias is not None:
            y = y + bias
        return y.reshape(x.shape[:-1] + (n,))
