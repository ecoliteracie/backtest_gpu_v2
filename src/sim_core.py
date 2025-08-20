# src/sim_core.py
from __future__ import annotations

from typing import Dict, Tuple, Any
import csv
from pathlib import Path
from datetime import datetime
import numpy as np

from .price_solver import find_price_for_target_rsi
from .gpu_events import NONE, BUY, SELL


def _symbol_from_cfg(cfg: dict) -> str:
    path = str(cfg.get("CSV_CACHE_FILE", "data.csv"))
    base = Path(path).stem
    if base:
        return base.split("_")[0]
    return "SYMBOL"


def simulate_once_from_events(
    df,
    rsi_maps: Dict[str, Dict[int, str]],
    events_type_1d: np.ndarray,   # (T,) int8 for one combo
    combo: Tuple[int, int, float, float],
    tolerance_pct: float,
    cfg: dict,
    regime_label: str,
    out_dir: str = "logs",
) -> Dict[str, Any]:
    """
    Price each BUY/SELL event for a single combo and write a trades CSV.
    Returns summary metrics and solver call counts.
    """
    buy_p, sell_p, buy_thr, sell_thr = combo
    low_map  = rsi_maps.get("low_map", {})
    high_map = rsi_maps.get("high_map", {})

    # File path
    symbol = _symbol_from_cfg(cfg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir_p / f"trades_{symbol}__{regime_label}__{buy_p}_{int(buy_thr)}_{sell_p}_{int(sell_thr)}__{ts}.csv"

    header = [
        "date", "action", "buy_period", "buy_thr", "sell_period", "sell_thr",
        "rsi_low_at_d", "rsi_high_at_d", "condition_met",
        "price", "mode", "reason",
        "low_price", "high_price"
    ]

    solver_stats = {
        "fast_buy": 0, "fast_sell": 0,
        "bsolve_buy": 0, "bsolve_sell": 0,
        "no_solution": 0
    }
    buy_cnt = 0
    sell_cnt = 0

    with trades_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for d, ev in enumerate(events_type_1d):
            if ev == NONE:
                continue

            date_str = str(df.index[d])
            if ev == BUY:
                side = "BUY"
                period = buy_p
                target = float(buy_thr)
                buy_cnt += 1
            elif ev == SELL:
                side = "SELL"
                period = sell_p
                target = float(sell_thr)
                sell_cnt += 1
            else:
                continue

            # Bound prices
            lo_price = float(df["Low"].iloc[d])
            hi_price = float(df["High"].iloc[d])

            # RSI bounds for the period (if available)
            rsi_low_col  = low_map.get(period, None)
            rsi_high_col = high_map.get(period, None)
            rsi_low_val  = float(df[rsi_low_col].iloc[d])  if rsi_low_col  and rsi_low_col in df.columns  else np.nan
            rsi_high_val = float(df[rsi_high_col].iloc[d]) if rsi_high_col and rsi_high_col in df.columns else np.nan

            # Fast-paths
            if side == "BUY" and np.isfinite(rsi_low_val) and np.isfinite(rsi_high_val):
                if (rsi_low_val < target) and (rsi_high_val < target):
                    w.writerow([date_str, side, buy_p, buy_thr, sell_p, sell_thr,
                                rsi_low_val, rsi_high_val, "Yes",
                                hi_price, "FAST_HIGH", "threshold already passed",
                                lo_price, hi_price])
                    solver_stats["fast_buy"] += 1
                    continue

            if side == "SELL" and np.isfinite(rsi_low_val) and np.isfinite(rsi_high_val):
                if (rsi_low_val > target) and (rsi_high_val > target):
                    w.writerow([date_str, side, buy_p, buy_thr, sell_p, sell_thr,
                                rsi_low_val, rsi_high_val, "Yes",
                                lo_price, "FAST_LOW", "threshold already passed",
                                lo_price, hi_price])
                    solver_stats["fast_sell"] += 1
                    continue

            # Binary-search style solve via proxy (RSI_LOW/HIGH interpolation)
            res = find_price_for_target_rsi(
                prev_closes=None,
                period=period,
                price_low=lo_price,
                price_high=hi_price,
                target_rsi=target,
                tolerance_pct=tolerance_pct,
                side=side,
                rsi_low_at_d=rsi_low_val if np.isfinite(rsi_low_val) else None,
                rsi_high_at_d=rsi_high_val if np.isfinite(rsi_high_val) else None,
            )

            if res["ok"]:
                mode = "BSOLVE"
                if side == "BUY":
                    solver_stats["bsolve_buy"] += 1
                else:
                    solver_stats["bsolve_sell"] += 1
                w.writerow([date_str, side, buy_p, buy_thr, sell_p, sell_thr,
                            rsi_low_val, rsi_high_val, "Yes",
                            float(res["price"]), mode, f"iters={res['iterations']}",
                            lo_price, hi_price])
            else:
                solver_stats["no_solution"] += 1
                w.writerow([date_str, side, buy_p, buy_thr, sell_p, sell_thr,
                            rsi_low_val, rsi_high_val, "No",    
                            "", "NO_SOLUTION", "unable to meet tolerance",
                            lo_price, hi_price])

    paired = min(buy_cnt, sell_cnt)
    return {
        "trades_path": str(trades_path),
        "buy_cnt": buy_cnt, "sell_cnt": sell_cnt, "paired": paired,
        "solver_stats": solver_stats,
    }
