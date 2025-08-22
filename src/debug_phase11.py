# src/debug_phase11.py
from __future__ import annotations

from typing import Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import csv
import re

from .gpu_events import build_event_streams, BUY, SELL, NONE
from .price_solver import find_price_for_target_rsi


def _safe_get(series, i) -> float:
    try:
        v = float(series.iloc[i])
        return v
    except Exception:
        return float("nan")


def _isfinite_all(*vals: float) -> bool:
    return all(np.isfinite(v) for v in vals)


def _label_for_filename(regime_label: Optional[str]) -> str:
    if not regime_label or regime_label == "gap_all":
        return "none-none"
    # sanitize e.g. "gap_(7.0,19.0)" -> "7.0-19.0", "gap_(None,-19.0)" -> "none--19.0"
    m = re.search(r"\(([^)]+)\)", regime_label)
    core = m.group(1) if m else regime_label
    return core.replace(", ", "_").replace(",", "-").replace(" ", "").replace("None", "none")


def emit_phase11_debug_csv(
    df,
    rsi_maps: Dict[str, Dict[int, str]],
    combo: Tuple[int, int, float, float],  # (buy_p, sell_p, buy_thr, sell_thr)
    regime_label: Optional[str],
    regime_mask_host: np.ndarray,          # bool mask aligned to df.index
    tolerance_pct: float,
    out_dir: str = "logs",
) -> str:
    """
    Produce a per-day audit CSV for the given combo and regime.

    Columns:
      date, rsi_low, rsi_high, rsi_close, price_low, price_high, price,
      is_buy_prev, is_buy_condi, is_buy,
      is_sell_prev, is_sell_condi, is_sell,
      regime_ok

    Notes:
      - rsi_* columns reflect the BUY period of the combo (for compactness).
      - is_buy / is_sell reflect the actual position-aware events from Phase 11's stream.
      - price is populated only on event days (BUY or SELL) using the same solver as Phase 11.
      - Booleans are left blank ("") if their inputs are not finite on that row.
    """
    buy_p, sell_p, buy_thr, sell_thr = combo
    close_map = rsi_maps.get("close_map", {})
    low_map   = rsi_maps.get("low_map", {})
    high_map  = rsi_maps.get("high_map", {})

    # Resolve RSI columns (BUY period shown in debug columns)
    rsi_close_buy_col = close_map.get(buy_p, None)
    rsi_low_buy_col   = low_map.get(buy_p, None)
    rsi_high_buy_col  = high_map.get(buy_p, None)

    # Also resolve SELL period RSI for pricing on sell days
    rsi_low_sell_col   = low_map.get(sell_p, None)
    rsi_high_sell_col  = high_map.get(sell_p, None)

    # Build a single-combo event stream using the current implementation (no behavior changes)
    streams = build_event_streams(
        df=df,
        rsi_maps=rsi_maps,
        regime_mask_host=regime_mask_host,
        combos=[combo],
        backend="auto",  # safe: will fallback to CPU if GPU kernel isn't supported
    )
    events_type = streams["events_type"][0]  # (T,) int8 for this combo

    # File path
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    tag = _label_for_filename(regime_label)
    fpath = out_dir_p / f"phase11_debug_{tag}.csv"

    header = [
        "date",
        "rsi_low", "rsi_high", "rsi_close",
        "price_low", "price_high",
        "price",
        "is_buy_prev", "is_buy_condi", "is_buy",
        "is_sell_prev", "is_sell_condi", "is_sell",
        "regime_ok",
    ]

    with fpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        n = len(df)
        for d in range(n):
            date_str = str(df.index[d])

            # Inputs
            lo_px = _safe_get(df["Low"], d)
            hi_px = _safe_get(df["High"], d)
            rsi_c_buy = _safe_get(df[rsi_close_buy_col], d) if rsi_close_buy_col else float("nan")
            rsi_l_buy = _safe_get(df[rsi_low_buy_col], d)   if rsi_low_buy_col   else float("nan")
            rsi_h_buy = _safe_get(df[rsi_high_buy_col], d)  if rsi_high_buy_col  else float("nan")

            # Previous CLOSE RSI (buy period) for "is_buy_prev"
            rsi_c_buy_prev = _safe_get(df[rsi_close_buy_col].shift(1), d) if rsi_close_buy_col else float("nan")

            # SELL-period bounds (for pricing sell events and sell "band" test)
            rsi_l_sell = _safe_get(df[rsi_low_sell_col], d)   if rsi_low_sell_col   else float("nan")
            rsi_h_sell = _safe_get(df[rsi_high_sell_col], d)  if rsi_high_sell_col  else float("nan")
            rsi_c_sell_prev = _safe_get(df[close_map.get(sell_p)].shift(1), d) if close_map.get(sell_p) else float("nan")

            # Regime mask flag
            regime_ok = bool(regime_mask_host[d]) if isinstance(regime_mask_host, np.ndarray) and len(regime_mask_host) == n else True

            # Helper booleans (blank if inputs are not finite)
            def _mk(val: bool, ok: bool) -> str | bool:
                return val if ok else ""

            # Buy helpers (using BUY period)
            prev_ok_buy = _isfinite_all(rsi_c_buy_prev) and (rsi_c_buy_prev < float(buy_thr))
            band_ok_buy = _isfinite_all(rsi_l_buy, rsi_h_buy) and ((rsi_l_buy <= float(buy_thr) <= rsi_h_buy) or (rsi_h_buy < float(buy_thr)))

            # Sell helpers (using SELL period)
            prev_ok_sell = _isfinite_all(rsi_c_sell_prev) and (rsi_c_sell_prev > float(sell_thr))
            band_ok_sell = _isfinite_all(rsi_l_sell, rsi_h_sell) and ((rsi_l_sell <= float(sell_thr) <= rsi_h_sell) or (rsi_l_sell > float(sell_thr)))

            # Actual event emitted by the current simulator
            ev = int(events_type[d])
            is_buy  = (ev == BUY)  and regime_ok
            is_sell = (ev == SELL) and regime_ok

            # Price: only compute when an event happened (use the correct period's bounds)
            out_price = ""
            if is_buy or is_sell:
                if is_buy:
                    rsi_l = rsi_l_buy
                    rsi_h = rsi_h_buy
                    target = float(buy_thr)
                else:
                    rsi_l = rsi_l_sell
                    rsi_h = rsi_h_sell
                    target = float(sell_thr)

                # FAST paths
                if _isfinite_all(rsi_l, rsi_h):
                    if is_buy and (rsi_l < target and rsi_h < target):
                        out_price = str(hi_px)
                    elif is_sell and (rsi_l > target and rsi_h > target):
                        out_price = str(lo_px)
                    else:
                        # Interpolate via solver (same as Phase 11)
                        res = find_price_for_target_rsi(
                            prev_closes=None,
                            period=buy_p if is_buy else sell_p,
                            price_low=lo_px,
                            price_high=hi_px,
                            target_rsi=target,
                            tolerance_pct=float(tolerance_pct),
                            side="BUY" if is_buy else "SELL",
                            rsi_low_at_d=rsi_l,
                            rsi_high_at_d=rsi_h,
                        )
                        if res.get("ok"):
                            out_price = str(float(res["price"]))
                        else:
                            out_price = ""  # no-solution leaves blank
                else:
                    out_price = ""  # insufficient inputs

            # Write the row
            w.writerow([
                date_str,
                rsi_l_buy if np.isfinite(rsi_l_buy) else "",
                rsi_h_buy if np.isfinite(rsi_h_buy) else "",
                rsi_c_buy if np.isfinite(rsi_c_buy) else "",
                lo_px if np.isfinite(lo_px) else "",
                hi_px if np.isfinite(hi_px) else "",
                out_price,
                _mk(prev_ok_buy, np.isfinite(rsi_c_buy_prev)),
                _mk(band_ok_buy,  _isfinite_all(rsi_l_buy, rsi_h_buy)),
                is_buy,
                _mk(prev_ok_sell, np.isfinite(rsi_c_sell_prev)),
                _mk(band_ok_sell, _isfinite_all(rsi_l_sell, rsi_h_sell)),
                is_sell,
                regime_ok,
            ])

    return str(fpath)
