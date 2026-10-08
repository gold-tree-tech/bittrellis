"""NVFP4 with GPTQ rounding of all three MLP projections, down_proj included, on the pinned calibration statistics.

`nvfp4_gptq` rounds gate_proj and up_proj by GPTQ on the pinned mean x xᵀ of the MLP input and gives down_proj
the `nvfp4_blockfit` MLP fit, because epoch hpc01-e6 pins no statistics for the down projection's input (the
SwiGLU activation). down_proj is a third of every MLP block's weights, and its input is the most heavy-tailed
activation in the block, so it is where Hessian-weighted rounding has the most error left to remove.

This encoder takes the Hessian of every MLP Linear from `input_hessian(ctx, lin)` and nothing else:

* a Linear with pinned statistics (gate/up now; down_proj as soon as an epoch pins its input) is rounded by
  the same GPTQ as `nvfp4_gptq`: columns left to right, each column's rounding error pushed through the
  inverse Hessian onto the columns not yet rounded, block scales chosen by the |e|^1.5 MLP fit as each
  16-column block is reached;
* a down_proj without pinned statistics gets the `nvfp4_blockfit` MLP fit, as in `nvfp4_gptq`. No statistic
  is estimated, approximated or synthesised here.

Differences from `nvfp4_gptq` beyond the down projection:

* the inverse-Hessian factor is cached per statistics tensor (gate/up share one, down_proj has its own), not
  per MLP block, so a block never reuses gate/up's factor for down_proj;
* the factor is computed with intermediates freed as it goes: the down_proj Hessian is [17408, 17408], and
  the float64 chain would otherwise hold five copies (about 12 GB) at once. The arithmetic is unchanged.

Determinism: NumPy on the CPU in float64, as `nvfp4_gptq` (fixed row chunks, fixed column block).
"""

from __future__ import annotations

import numpy as np

from ..precision import NVFP4
from . import nvfp4_blockfit as bf
from . import nvfp4_gptq as gq
from .base import Produced, QuantContext, Quantizer, f32_weight, input_hessian

DAMP = gq.DAMP


def inverse_hessian_factor(h: np.ndarray, damp: float = DAMP) -> np.ndarray:
    """Upper Cholesky factor U of (H + λI)⁻¹ = UᵀU in float64; `nvfp4_gptq.inverse_hessian_factor` with
    each intermediate released as soon as it is used (same operations, same bytes)."""
    h = np.array(h, dtype=np.float64)
    d = np.diag(h).copy()
    dead = d <= 0
    h[dead, dead] = 1.0
    h[np.diag_indices_from(h)] += damp * float(np.mean(np.diag(h)))
    lo = np.linalg.cholesky(h)
    del h
    inv = np.linalg.inv(lo)
    del lo
    hinv = inv.T @ inv
    del inv
    return np.linalg.cholesky(hinv).T


def quantize(w: np.ndarray, h: np.ndarray, amax: float | None = None) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """float32 [rows, cols] and H [cols, cols] -> ModelOpt NVFP4 (packed, block scales, weight_scale_2)."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % 16 or not np.isfinite(w).all():
        raise ValueError("expected finite [rows, cols] with cols divisible by 16")
    h = np.asarray(h)
    if h.shape != (w.shape[1], w.shape[1]) or not np.isfinite(h).all():
        raise ValueError("expected a finite [cols, cols] input Hessian")
    return gq._quantize_with(w, inverse_hessian_factor(h), amax)


def _stat_key(lin) -> str:
    from ..calibration import stat_name

    return stat_name(lin.prefix) or lin.prefix


class NVFP4GPTQDown(Quantizer):
    name = "nvfp4_gptq_dn"
    version = 1
    formats = (NVFP4,)
    lineage = "regenerable"
    replay_mode = "independent"
    description = "NVFP4 MLP: GPTQ rounding of gate/up and of down_proj on the pinned calibration statistics (blockfit for down without them)"

    def supports(self, unit, fmt: str) -> bool:
        return fmt == NVFP4 and unit.kind == "mlp"

    def available(self, ctx: QuantContext, unit, fmt: str) -> str | None:
        return None if ctx.calibration is not None else "needs the calibration statistics (bittrellis calibration fetch)"

    def encode(self, ctx: QuantContext, unit, lin, fmt: str) -> list[Produced]:
        if not self.supports(unit, fmt):
            raise ValueError(f"{self.ref} encodes MLP blocks as NVFP4 only")
        if ctx.calibration is None:
            raise ValueError(f"{self.ref} needs the calibration statistics (bittrellis calibration fetch)")
        h = input_hessian(ctx, lin)
        if h is None:
            if not lin.prefix.endswith(".down_proj"):
                raise ValueError(f"{lin.prefix}: the pinned calibration has no statistics for this Linear")
            return bf.NVFP4BlockFit().encode(ctx, unit, lin, fmt)
        cache = ctx.state.setdefault(self.name, {})
        key = _stat_key(lin)
        if cache.get("key") != key:
            cache.pop("u", None)                      # free the previous factor before building the next
            cache["key"], cache["u"] = key, inverse_hessian_factor(h)
        w = f32_weight(ctx, lin)
        if not np.isfinite(w).all():
            raise ValueError(f"{lin.prefix}: base weight is not finite")
        packed, scales, ws2 = gq._quantize_with(w, cache["u"])
        return [(".weight", "U8", packed.shape, packed),
                (".weight_scale", "F8_E4M3", scales.shape, scales),
                (".weight_scale_2", "F32", (), np.asarray(ws2, "<f4").reshape(()))]
