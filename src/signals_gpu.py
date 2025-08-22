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
    low_map: Dict[int, str],
    high_map: Dict[int, str],
    buy_period: int,
    sell_period: int,
    buy_thr: float,
    sell_thr: float,
    regime_mask_host: Optional[np.ndarray] = None,
    cp=None,  # may be a CuPy module OR a backend dict from select_backend
) -> Tuple["cp.ndarray", "cp.ndarray", Dict]:
    """
    Build GPU buy/sell predicate masks using:
      BUY : RSI_CLOSE[d-1] < buy_thr AND (band brackets buy_thr OR band entirely below buy_thr)
      SELL: RSI_CLOSE[d-1] > sell_thr AND (band brackets sell_thr OR band entirely above sell_thr)
    No momentum; day 0 is always False.
    """
    cp, backend_meta = _resolve_backend_cp(cp)

    # Sanity on periods
    for p, m, name in [
        (buy_period, close_map, "close_map(buy)"),
        (sell_period, close_map, "close_map(sell)"),
        (buy_period, low_map,   "low_map(buy)"),
        (buy_period, high_map,  "high_map(buy)"),
        (sell_period, low_map,  "low_map(sell)"),
        (sell_period, high_map, "high_map(sell)"),
    ]:
        if p not in m:
            raise ValueError(f"RSI period {p} not present in {name}.")

    # If thresholds inverted, return empty masks with reason
    if buy_thr >= sell_thr:
        n = len(df)
        empty = cp.zeros(n, dtype=cp.bool_)
        regime_true = int(regime_mask_host.sum()) if isinstance(regime_mask_host, np.ndarray) else n
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

    # Column names
    rsi_close_buy_col = close_map[buy_period]
    rsi_low_buy_col   = low_map[buy_period]
    rsi_high_buy_col  = high_map[buy_period]

    rsi_close_sell_col = close_map[sell_period]
    rsi_low_sell_col   = low_map[sell_period]
    rsi_high_sell_col  = high_map[sell_period]

    # Host->device transfers (float64)
    rsi_c_buy  = cp.asarray(df[rsi_close_buy_col].to_numpy(np.float64, copy=False))
    rsi_c_sell = cp.asarray(df[rsi_close_sell_col].to_numpy(np.float64, copy=False))

    rsi_l_buy  = cp.asarray(df[rsi_low_buy_col].to_numpy(np.float64, copy=False))
    rsi_h_buy  = cp.asarray(df[rsi_high_buy_col].to_numpy(np.float64, copy=False))
    rsi_l_sell = cp.asarray(df[rsi_low_sell_col].to_numpy(np.float64, copy=False))
    rsi_h_sell = cp.asarray(df[rsi_high_sell_col].to_numpy(np.float64, copy=False))

    # Previous-day arrays (device)
    rsi_c_buy_prev  = _shift_dev(cp, rsi_c_buy,  fill_value=cp.nan)
    rsi_c_sell_prev = _shift_dev(cp, rsi_c_sell, fill_value=cp.nan)

    # Regime mask on device
    n = rsi_c_buy.shape[0]
    if regime_mask_host is None:
        regime_mask_dev = cp.ones(n, dtype=cp.bool_)
        regime_true_count = int(n)
        regime_label = None
    else:
        if len(regime_mask_host) != n:
            raise ValueError(f"regime_mask_host length {len(regime_mask_host)} does not match df length {n}.")
        regime_mask_dev = cp.asarray(regime_mask_host.astype(bool, copy=False))
        regime_true_count = int(np.count_nonzero(regime_mask_host))
        regime_label = None

    # NaN validity masks
    isfinite = cp.isfinite
    valid_buy  = isfinite(rsi_c_buy_prev)  & isfinite(rsi_l_buy)  & isfinite(rsi_h_buy)
    valid_sell = isfinite(rsi_c_sell_prev) & isfinite(rsi_l_sell) & isfinite(rsi_h_sell)

    # Band checks (device)
    # BUY: bracket OR entirely below threshold
    buy_band = ((rsi_l_buy <= buy_thr) & (buy_thr <= rsi_h_buy)) | (rsi_h_buy < buy_thr)
    # SELL: bracket OR entirely above threshold
    sell_band = ((rsi_l_sell <= sell_thr) & (sell_thr <= rsi_h_sell)) | (rsi_l_sell > sell_thr)

    # Predicates
    buy_ok  = (rsi_c_buy_prev  < buy_thr) & buy_band
    sell_ok = (rsi_c_sell_prev > sell_thr) & sell_band

    # Apply validity + regime
    buy_ok  &= valid_buy  & regime_mask_dev
    sell_ok &= valid_sell & regime_mask_dev

    # Day-0 must be False
    if n > 0:
        buy_ok[0] = False
        sell_ok[0] = False

    # Counts and sample indices (host)
    buy_count  = int(cp.count_nonzero(buy_ok).item())
    sell_count = int(cp.count_nonzero(sell_ok).item())

    def first_dates(mask_dev, k=5) -> List[str]:
        idx_dev = cp.where(mask_dev)[0]
        if idx_dev.size == 0:
            return []
        idx_host = cp.asnumpy(idx_dev[:k]).tolist()
        return [str(df.index[i]) for i in idx_host]

    samples_buy  = first_dates(buy_ok,  k=5)
    samples_sell = first_dates(sell_ok, k=5)

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