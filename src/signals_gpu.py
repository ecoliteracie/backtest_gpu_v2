# src/signals_gpu.py
from __future__ import annotations

from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import pandas as pd


def _device_info_from_cp(cp) -> Dict[str, str]:
    """Query device name/cc from a CuPy module."""
    try:
        dev_id = cp.cuda.runtime.getDevice()
        props = cp.cuda.runtime.getDeviceProperties(dev_id)
        name = props.get("name", b"").decode("utf-8") if isinstance(props.get("name"), (bytes, bytearray)) else str(props.get("name"))
        major = props.get("major", 0)
        minor = props.get("minor", 0)
        return {"name": name, "cc": f"{major}.{minor}", "id": str(dev_id)}
    except Exception:
        return {"name": "Unknown", "cc": "?", "id": "?"}


def _resolve_backend_cp(cp_or_backend: Any):
    """
    Accept either:
      - a CuPy module, or
      - a backend dict returned by gpu_backend.select_backend("cupy") that includes {"xp": cp, ...}
      - None (we'll call select_backend ourselves)
    Returns: (cp_module, backend_meta_dict)
    """
    if cp_or_backend is None:
        from gpu_backend import select_backend  # local import
        backend = select_backend("cupy")
        return backend["xp"], backend

    # If caller passed the backend dict directly
    if isinstance(cp_or_backend, dict) and "xp" in cp_or_backend:
        return cp_or_backend["xp"], cp_or_backend

    # Assume it's a CuPy module
    return cp_or_backend, None


def _shift_dev(cp, arr: "cp.ndarray", fill_value=np.nan) -> "cp.ndarray":
    """Device-side shift(1) equivalent: prepend fill_value, drop last."""
    out = cp.empty_like(arr)
    out[0] = fill_value
    out[1:] = arr[:-1]
    return out


def make_buy_sell_masks(
    df: pd.DataFrame,
    close_map: Dict[int, str],
    buy_period: int,
    sell_period: int,
    buy_thr: float,
    sell_thr: float,
    regime_mask_host: Optional[np.ndarray] = None,
    cp=None,  # may be a CuPy module OR a backend dict from select_backend
) -> Tuple["cp.ndarray", "cp.ndarray", Dict]:
    """
    Build GPU buy/sell predicate masks using CLOSE-based RSI and two-day rules.

    Returns:
      buy_ok_dev (cp.ndarray bool), sell_ok_dev (cp.ndarray bool), meta (dict)
    """
    cp, backend_meta = _resolve_backend_cp(cp)

    # Sanity on periods
    if buy_period not in close_map:
        raise ValueError(f"buy_period {buy_period} not present in close_map.")
    if sell_period not in close_map:
        raise ValueError(f"sell_period {sell_period} not present in close_map.")

    # If thresholds inverted, return empty masks with reason
    if buy_thr >= sell_thr:
        n = len(df)
        empty = cp.zeros(n, dtype=cp.bool_)
        # regime size
        regime_true = int(regime_mask_host.sum()) if isinstance(regime_mask_host, np.ndarray) else n
        # backend info
        backend_info = (
            {"name": backend_meta.get("device_name"), "cc": backend_meta.get("cc")}
            if isinstance(backend_meta, dict)
            else _device_info_from_cp(cp)
        )
        meta = {
            "params": {"buy_period": buy_period, "sell_period": sell_period, "buy_thr": buy_thr, "sell_thr": sell_thr},
            "counts": {"buy_ok": 0, "sell_ok": 0},
            "first_true_dates": {"buy": [], "sell": []},
            "regime": {"applied": None, "true_count": regime_true},
            "backend": backend_info,
            "reason": "buy_thr >= sell_thr",
        }
        return empty, empty, meta

    # Resolve series
    rsi_buy_col = close_map[buy_period]
    rsi_sell_col = close_map[sell_period]

    # Host->device transfers (float64)
    close_h = df["Close"].to_numpy(np.float64, copy=False)
    rsi_buy_h = df[rsi_buy_col].to_numpy(np.float64, copy=False)
    rsi_sell_h = df[rsi_sell_col].to_numpy(np.float64, copy=False)

    close = cp.asarray(close_h)
    rsi_buy = cp.asarray(rsi_buy_h)
    rsi_sell = cp.asarray(rsi_sell_h)

    # Previous day arrays (device)
    close_prev = _shift_dev(cp, close, fill_value=cp.nan)
    rsi_buy_prev = _shift_dev(cp, rsi_buy, fill_value=cp.nan)
    rsi_sell_prev = _shift_dev(cp, rsi_sell, fill_value=cp.nan)

    # Regime mask on device
    n = close.shape[0]
    if regime_mask_host is None:
        regime_mask_dev = cp.ones(n, dtype=cp.bool_)
        regime_true_count = int(n)
        regime_label = None
    else:
        if len(regime_mask_host) != n:
            raise ValueError(f"regime_mask_host length {len(regime_mask_host)} does not match df length {n}.")
        regime_mask_dev = cp.asarray(regime_mask_host.astype(bool, copy=False))
        regime_true_count = int(np.count_nonzero(regime_mask_host))
        regime_label = None  # caller can overwrite later

    # NaN validity masks
    isfinite = cp.isfinite
    valid_buy = isfinite(rsi_buy_prev) & isfinite(rsi_buy) & isfinite(close_prev) & isfinite(close)
    valid_sell = isfinite(rsi_sell_prev) & isfinite(rsi_sell)

    # Two-day predicates (device, vectorized)
    buy_ok = (rsi_buy_prev < buy_thr) & (rsi_buy < buy_thr) & (close > close_prev)
    sell_ok = (rsi_sell_prev > sell_thr) & (rsi_sell > sell_thr)

    # Apply validity and regime
    buy_ok &= valid_buy & regime_mask_dev
    sell_ok &= valid_sell & regime_mask_dev

    # Day-0 must be False
    if n > 0:
        buy_ok[0] = False
        sell_ok[0] = False

    # Counts and sample indices (host)
    buy_count = int(cp.count_nonzero(buy_ok).item())
    sell_count = int(cp.count_nonzero(sell_ok).item())

    def first_dates(mask_dev, k=5) -> List[str]:
        idx_dev = cp.where(mask_dev)[0]
        if idx_dev.size == 0:
            return []
        idx_host = cp.asnumpy(idx_dev[:k]).tolist()
        return [str(df.index[i]) for i in idx_host]

    samples_buy = first_dates(buy_ok, k=5)
    samples_sell = first_dates(sell_ok, k=5)

    # Backend info for banner
    backend_info = (
        {"name": backend_meta.get("device_name"), "cc": backend_meta.get("cc")}
        if isinstance(backend_meta, dict)
        else _device_info_from_cp(cp)
    )

    meta = {
        "params": {"buy_period": buy_period, "sell_period": sell_period, "buy_thr": buy_thr, "sell_thr": sell_thr},
        "counts": {"buy_ok": buy_count, "sell_ok": sell_count},
        "first_true_dates": {"buy": samples_buy, "sell": samples_sell},
        "regime": {"applied": regime_label, "true_count": regime_true_count},
        "backend": backend_info,
    }
    return buy_ok, sell_ok, meta
