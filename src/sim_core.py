# --- add/replace in src/sim_core.py ---
from __future__ import annotations

import os
import math
import csv
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# Expect these to be present from your Phase-7 output:
# rsi_maps = {"close_map": {p: col}, "low_map": {p: col}, "high_map": {p: col}}

BUY, SELL = 1, 2  # keep consistent with gpu_events

def _infer_symbol_from_cfg(cfg: Dict[str, Any]) -> str:
    path = str(cfg.get("CSV_CACHE_FILE", "") or "")
    base = os.path.basename(path)
    # Heuristic: SOXL_full_ohlc_indicators.csv -> SOXL
    if "_" in base:
        return base.split("_", 1)[0].upper()
    if "." in base:
        return base.split(".", 1)[0].upper()
    return (base or "SYMBOL").upper()

def _regime_tag(label: str | None) -> str:
    if label in (None, "gap_all"):
        return "gap_all"
    # Example input: "gap_(None,-19.0)" -> "gap_-inf_-19.0"
    tag = str(label).strip()
    tag = tag.replace("gap_(", "").replace(")", "")
    tag = tag.replace("None", "inf")
    # swap leading 'inf' to -inf when it was (None, x)
    if tag.startswith("inf,"):
        tag = tag.replace("inf,", "-inf_", 1)
    else:
        tag = tag.replace(",", "_")
    tag = tag.replace(" ", "")
    tag = f"gap_{tag}"
    return tag

def _years_between(first_ts: pd.Timestamp, last_ts: pd.Timestamp) -> float:
    days = (last_ts - first_ts).days
    return max(days, 1) / 365.25

def _compute_roi_cagr(final_value: float, total_invested: float, first_dt: pd.Timestamp, last_dt: pd.Timestamp) -> Tuple[float, float]:
    if total_invested <= 0:
        return 0.0, 0.0
    roi = (final_value / total_invested - 1.0) * 100.0
    yrs = _years_between(first_dt, last_dt)
    try:
        cagr = (final_value / total_invested) ** (1.0 / yrs) - 1.0
        cagr *= 100.0
    except Exception:
        cagr = 0.0
    return roi, cagr

def _price_for_event(
    df: pd.DataFrame,
    rsi_maps: Dict[str, Dict[int, str]],
    day_idx: int,
    side: int,
    period: int,
    thr: float,
    tolerance_pct: float,
) -> Tuple[float | None, str, Dict[str, float]]:
    """
    Return (price, mode, extras) where extras has rsi_low_at_d, rsi_high_at_d, rsi_close_at_d, low_price, high_price.
    """
    low_col  = rsi_maps["low_map"][period]
    high_col = rsi_maps["high_map"][period]
    close_col = rsi_maps["close_map"][period]

    rsi_low_at_d  = float(df[low_col].iloc[day_idx])
    rsi_high_at_d = float(df[high_col].iloc[day_idx])
    rsi_close_at_d = float(df[close_col].iloc[day_idx])
    low_price   = float(df["Low"].iloc[day_idx])
    high_price  = float(df["High"].iloc[day_idx])

    # Fast paths first
    if side == BUY:
        # whole band already below buy threshold -> take day's High
        if (np.isfinite(rsi_low_at_d) and np.isfinite(rsi_high_at_d)) and (rsi_low_at_d < thr and rsi_high_at_d < thr):
            return high_price, "FAST_HIGH", {
                "rsi_low_at_d": rsi_low_at_d, "rsi_high_at_d": rsi_high_at_d, "rsi_close_at_d": rsi_close_at_d,
                "low_price": low_price, "high_price": high_price
            }
    else:  # SELL
        # whole band already above sell threshold -> take day's Low
        if (np.isfinite(rsi_low_at_d) and np.isfinite(rsi_high_at_d)) and (rsi_low_at_d > thr and rsi_high_at_d > thr):
            return low_price, "FAST_LOW", {
                "rsi_low_at_d": rsi_low_at_d, "rsi_high_at_d": rsi_high_at_d, "rsi_close_at_d": rsi_close_at_d,
                "low_price": low_price, "high_price": high_price
            }

    # Otherwise binary search using your solver (bracketing case)
    try:
        from src.price_solver import find_price_for_target_rsi  # your existing function
        # Build prev closes for this period
        start = max(0, day_idx - period)
        prev_closes = df["Close"].iloc[start:day_idx].to_numpy(np.float64, copy=False)
        res = find_price_for_target_rsi(
            prev_closes=prev_closes,
            period=period,
            price_low=low_price,
            price_high=high_price,
            target_rsi=float(thr),
            tolerance_pct=float(tolerance_pct),
            side=("BUY" if side == BUY else "SELL"),
        )
        if isinstance(res, dict):
            price = res.get("price", None)
            mode = res.get("mode", "BSOLVE")
        else:
            price = float(res) if res is not None else None
            mode = "BSOLVE"
    except Exception:
        price, mode = None, "NO_SOLUTION"

    return price, mode, {
        "rsi_low_at_d": rsi_low_at_d, "rsi_high_at_d": rsi_high_at_d, "rsi_close_at_d": rsi_close_at_d,
        "low_price": low_price, "high_price": high_price
    }

def simulate_once_from_events(
    df: pd.DataFrame,
    rsi_maps: Dict[str, Dict[int, str]],
    events_type_1d: np.ndarray,  # shape (T,), values 0/1/2
    combo: Tuple[int, int, float, float],  # (buy_p, sell_p, buy_thr, sell_thr)
    tolerance_pct: float,
    cfg: Dict[str, Any],
    regime_label: str | None,
    out_dir: str = "logs",
    write_trades: bool = True,   # NEW: allow silent run when summarizing many combos
) -> Dict[str, Any]:
    """
    Prices all BUY/SELL events in events_type_1d, simulates a single-position strategy,
    and (optionally) writes a trades CSV. Returns summary stats.
    """
    os.makedirs(out_dir, exist_ok=True)
    T = len(df)
    buy_p, sell_p, buy_thr, sell_thr = combo

    # Portfolio parameters
    initial_cash = float(cfg.get("INITIAL_CASH", 1000.0))
    daily_cash   = float(cfg.get("DAILY_CASH", 0.0))
    cash = initial_cash
    shares = 0.0

    # Accounting
    buy_cnt = 0
    sell_cnt = 0
    solver_calls = {"FAST_BUY": 0, "FAST_SELL": 0, "BSOLVE_BUY": 0, "BSOLVE_SELL": 0, "NO_SOLUTION": 0}

    # Trades CSV path
    symbol = _infer_symbol_from_cfg(cfg)
    tag = _regime_tag(regime_label)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = os.path.join(out_dir, f"trades_{symbol}__{tag}__{buy_p}_{int(buy_thr)}_{sell_p}_{int(sell_thr)}__{ts}.csv")

    # Prepare writer (only if writing)
    writer = None
    f = None
    if write_trades:
        f = open(trades_path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow([
            "date","action","buy_period","buy_thr","sell_period","sell_thr",
            "rsi_low_at_d","rsi_high_at_d","rsi_close_at_d","condition_met","price","mode","reason",
            "low_price","high_price"
        ])

    in_pos = False
    last_close = float(df["Close"].iloc[-1])

    for d in range(T):
        # Daily injection at start of day
        cash += daily_cash

        evt = int(events_type_1d[d])
        if evt not in (BUY, SELL):
            continue

        # Determine side & params
        side = evt
        period = buy_p if side == BUY else sell_p
        thr    = buy_thr if side == BUY else sell_thr

        # Pricing
        price, mode, extras = _price_for_event(
            df=df, rsi_maps=rsi_maps, day_idx=d, side=side, period=period, thr=thr, tolerance_pct=tolerance_pct
        )

        # Condition met if we actually got a price
        condition_met = (price is not None) and np.isfinite(price)

        # Update solver stats
        if mode == "FAST_HIGH":
            solver_calls["FAST_BUY"] += 1
        elif mode == "FAST_LOW":
            solver_calls["FAST_SELL"] += 1
        elif mode == "BSOLVE":
            if side == BUY:
                solver_calls["BSOLVE_BUY"] += 1
            else:
                solver_calls["BSOLVE_SELL"] += 1
        else:  # "NO_SOLUTION"
            solver_calls["NO_SOLUTION"] += 1

        # Position logic (single position)
        reason = ""
        if side == BUY:
            if in_pos:
                reason = "already_long"
            elif not condition_met:
                reason = "no_price"
            else:
                # buy all-in
                if price > 0.0 and cash > 0.0:
                    shares = cash / price
                    cash = 0.0
                    in_pos = True
                    buy_cnt += 1
                else:
                    reason = "bad_price_or_cash"
        else:  # SELL
            if not in_pos:
                reason = "not_long"
            elif not condition_met:
                reason = "no_price"
            else:
                # sell all
                if price > 0.0 and shares > 0.0:
                    cash += shares * price
                    shares = 0.0
                    in_pos = False
                    sell_cnt += 1
                else:
                    reason = "bad_price_or_shares"

        # Write one trade row if requested
        if write_trades and condition_met:
            writer.writerow([
                str(df.index[d]), ("BUY" if side == BUY else "SELL"),
                buy_p, buy_thr, sell_p, sell_thr,
                extras["rsi_low_at_d"], extras["rsi_high_at_d"], extras["rsi_close_at_d"],
                "Yes" if condition_met else "No",
                ("" if price is None else price),
                mode,
                reason,
                extras["low_price"], extras["high_price"],
            ])

    # Final portfolio
    final_value = float(cash + shares * last_close)
    total_invested = float(initial_cash + daily_cash * T)
    roi_pct, cagr_pct = _compute_roi_cagr(final_value, total_invested, df.index[0], df.index[-1])

    if write_trades and f is not None:
        try:
            f.flush()
        finally:
            f.close()

    return {
        "final_value": final_value,
        "buy_cnt": buy_cnt,
        "sell_cnt": sell_cnt,
        "invested_cash": total_invested,
        "roi_pct": roi_pct,
        "cagr_pct": cagr_pct,
        "solver_stats": solver_calls,
        "trades_path": trades_path if write_trades else "",
    }

def simulate_grid_summary(
    df: pd.DataFrame,
    rsi_maps: Dict[str, Dict[int, str]],
    events_type: np.ndarray,                        # shape (C, T)
    combos: List[Tuple[int, int, float, float]],    # length C
    cfg: Dict[str, Any],
    regime_label: str | None,
    out_dir: str = "results",
) -> str:
    """
    Run pricing + single-position accounting for every combo, collect summary rows,
    and write one CSV per regime at results/<symbol>__<regime>.csv.
    Returns the CSV path.
    """
    os.makedirs(out_dir, exist_ok=True)
    T = len(df)
    symbol = _infer_symbol_from_cfg(cfg)
    tag = _regime_tag(regime_label)
    out_path = os.path.join(out_dir, f"summary_{symbol}__{tag}.csv")

    tol_pct = float(cfg.get("RSI_TOLERANCE_PCT", 0.001))

    rows = []
    for i, combo in enumerate(combos):
        ev = events_type[i]
        res = simulate_once_from_events(
            df=df,
            rsi_maps=rsi_maps,
            events_type_1d=ev,
            combo=combo,
            tolerance_pct=tol_pct,
            cfg=cfg,
            regime_label=regime_label,
            out_dir="logs",         # keep detailed trades in logs/ if you want; or set write_trades=False
            write_trades=False,     # summary run: no per-trade CSV
        )
        buy_p, sell_p, buy_thr, sell_thr = combo
        rows.append([
            buy_p, sell_p, buy_thr, sell_thr,
            res["final_value"], res["roi_pct"], res["cagr_pct"],
            res["buy_cnt"], res["sell_cnt"], res["invested_cash"],
        ])

    # Write summary table
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Buy RSI Period", "Sell RSI Period", "Buy RSI <", "Sell RSI >",
            "Swing Final Portfolio", "Swing ROI %", "Swing CAGR %",
            "Buy Count", "Sell Count", "Total Invested"
        ])
        w.writerows(rows)

    return out_path
