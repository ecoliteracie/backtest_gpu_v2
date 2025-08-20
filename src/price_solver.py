# src/price_solver.py
from __future__ import annotations

from typing import Dict, Optional
import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def find_price_for_target_rsi(
    prev_closes: Optional[np.ndarray],
    period: int,
    price_low: float,
    price_high: float,
    target_rsi: float,
    tolerance_pct: float,
    side: str,
    rsi_low_at_d: Optional[float] = None,
    rsi_high_at_d: Optional[float] = None,
    max_iter: int = 20,
) -> Dict[str, object]:
    """
    Estimate the bar price in [Low, High] at which RSI(period) equals target_rsi.

    Practical approach:
      - If RSI_LOW/RSI_HIGH at day d are available and finite, use monotone linear
        interpolation as the function proxy (fast and consistent with precomputed series).
      - Else fall back to a simple bisection over [Low, High] using the same linear proxy.

    Returns:
      {
        "ok": bool, "price": float or None, "mode": "FAST"|"BSOLVE"|"NO_SOLUTION",
        "iterations": int, "est_rsi": float or None
      }
    """
    lo = float(price_low)
    hi = float(price_high)
    if not (np.isfinite(lo) and np.isfinite(hi)) or not (hi >= lo):
        return {"ok": False, "price": None, "mode": "NO_SOLUTION", "iterations": 0, "est_rsi": None}

    # Absolute tolerance in RSI points (tolerance_pct is given as percent-of-100)
    tol_rsi = float(tolerance_pct)

    # If both bounds already on one side, the caller should have short-circuited.
    # Here we handle the in-between case via linear interpolation of RSI vs price.
    if (rsi_low_at_d is not None) and (rsi_high_at_d is not None) and \
       np.isfinite(rsi_low_at_d) and np.isfinite(rsi_high_at_d) and (hi > lo):
        rlo = float(rsi_low_at_d)
        rhi = float(rsi_high_at_d)

        if rhi == rlo:
            # Flat RSI across the bar; any price works numerically—return midpoint.
            mid = 0.5 * (lo + hi)
            return {"ok": True, "price": mid, "mode": "BSOLVE", "iterations": 0, "est_rsi": rlo}

        # Linear map: r(p) = rlo + (rhi-rlo)*((p-lo)/(hi-lo))
        est = lo + (hi - lo) * ((target_rsi - rlo) / (rhi - rlo))
        est = _clamp(est, lo, hi)
        # Check residual using the same proxy
        est_rsi = rlo + (rhi - rlo) * ((est - lo) / (hi - lo))
        if abs(est_rsi - target_rsi) <= tol_rsi:
            return {"ok": True, "price": est, "mode": "BSOLVE", "iterations": 1, "est_rsi": est_rsi}

        # If not within tolerance, refine with bisection on the proxy
        left, right = lo, hi
        it = 1
        for _ in range(max_iter):
            it += 1
            mid = 0.5 * (left + right)
            mid_rsi = rlo + (rhi - rlo) * ((mid - lo) / (hi - lo))
            if abs(mid_rsi - target_rsi) <= tol_rsi:
                return {"ok": True, "price": mid, "mode": "BSOLVE", "iterations": it, "est_rsi": mid_rsi}
            # Decide which half to keep by monotonicity of the proxy
            if (mid_rsi < target_rsi and rhi >= rlo) or (mid_rsi > target_rsi and rhi < rlo):
                left = mid
            else:
                right = mid
        # Could not satisfy tolerance
        return {"ok": False, "price": None, "mode": "NO_SOLUTION", "iterations": it, "est_rsi": mid_rsi}

    # Fallback: no RSI bounds—return midpoint as a last resort
    mid = 0.5 * (lo + hi)
    return {"ok": False, "price": mid, "mode": "NO_SOLUTION", "iterations": 0, "est_rsi": None}
