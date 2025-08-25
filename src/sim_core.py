# --- add/replace in src/sim_core.py ---
from __future__ import annotations

import csv
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import os, math, itertools, time
from pathlib import Path

from src.gpu_events import build_event_streams
from src.price_solver import find_price_for_target_rsi
from src.banners import phase11 as banner11_mod 
from src.logging_setup import get_logger


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

# --- sim_core.py (replace simulate_grid_summary entirely) ---
# Reuse your existing helpers in this file:
# - simulate_once_from_events (unchanged)
# If you don’t have it, keep your current version; this function calls it.

def _safe_label(label: str | None) -> str:
    if label in (None, "gap_all", "gap_(None,None)"):
        return "gap_all"
    s = str(label)
    s = s.replace("(", "").replace(")", "").replace(" ", "")
    s = s.replace(",", "_").replace("None", "inf")
    s = s.replace("-inf", "neginf")  # prevent double minus sequences in filenames
    s = s.replace("__", "_")
    return s

def _int_range_inclusive(lo: int, hi: int) -> List[int]:
    if lo > hi:
        lo, hi = hi, lo
    return list(range(int(lo), int(hi) + 1))

def _int_range_exclusive_max(lo: int, hi_exclusive: int) -> List[int]:
    if lo >= hi_exclusive:
        return []
    return list(range(int(lo), int(hi_exclusive)))

def _grid_from_cfg(cfg: Dict) -> Tuple[List[int], List[int], List[int], List[int]]:
    # Periods: expected 2..10 inclusive (step=1), per your acceptance math (9 values).
    rlo = int(cfg.get("RSI_MIN", 2))
    rhi = int(cfg.get("RSI_MAX", 14))
    buy_periods  = _int_range_inclusive(rlo, rhi)
    sell_periods = _int_range_inclusive(rlo, rhi)

    # Thresholds: MAX is exclusive per your expectation (20..39) and (50..89).
    bt_lo = int(cfg.get("BUY_THRESHOLD_MIN", 20))
    bt_hi = int(cfg.get("BUY_THRESHOLD_MAX", 40))  # exclusive
    st_lo = int(cfg.get("SELL_THRESHOLD_MIN", 60))
    st_hi = int(cfg.get("SELL_THRESHOLD_MAX", 90))  # exclusive
    buy_thrs  = _int_range_exclusive_max(bt_lo, bt_hi)
    sell_thrs = _int_range_exclusive_max(st_lo, st_hi)

    return buy_periods, sell_periods, buy_thrs, sell_thrs

def _expected_combo_count(bp: List[int], sp: List[int], bt: List[int], st: List[int]) -> int:
    return len(bp) * len(sp) * len(bt) * len(st)

def simulate_grid_summary(
    df: pd.DataFrame,
    rsi_maps: Dict[str, Dict[int, str]],
    regime_label: str | None,
    cfg: Dict,
    out_dir: str = "results",
    backend: str = "auto",
    batch_size: int | None = None,
) -> str:
    """
    Runs the full grid from cfg and writes one CSV with summary rows for the given regime.
    Returns the output CSV path.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger("phase11_summary", Path("logs") / "phase11_summary.log")

    # Build full ranges (no sampling)
    buy_periods, sell_periods, buy_thrs, sell_thrs = _grid_from_cfg(cfg)

    # Diagnostics
    exp = _expected_combo_count(buy_periods, sell_periods, buy_thrs, sell_thrs)
    hdr = (
        f"[GRID] buy_p={buy_periods[0]}..{buy_periods[-1]}({len(buy_periods)}), "
        f"sell_p={sell_periods[0]}..{sell_periods[-1]}({len(sell_periods)}), "
        f"buy_thr={buy_thrs[0]}..{buy_thrs[-1]}({len(buy_thrs)} excl-max), "
        f"sell_thr={sell_thrs[0]}..{sell_thrs[-1]}({len(sell_thrs)} excl-max)"
    )
    print(hdr); logger.info(hdr)
    print(f"[EXPECTED] combos={exp}"); logger.info(f"[EXPECTED] combos={exp}")

    # Build the all-True or per-regime mask
    from src import regimes
    if regime_label in (None, "gap_all", "gap_(None,None)"):
        regime_mask_host = np.ones(len(df), dtype=bool)
    else:
        regime_mask_host = regimes.regime_mask(df, regime_label)

    # Compose the combos (skip bt >= st)
    combos: List[Tuple[int, int, float, float]] = []
    for bp in buy_periods:
        for sp in sell_periods:
            for bt in buy_thrs:
                for st in sell_thrs:
                    if bt < st:
                        combos.append((int(bp), int(sp), float(bt), float(st)))

    if len(combos) != exp:
        msg = f"[WARN] filtered (bt<st) combos={len(combos)} differs from expected={exp}"
        print(msg); logger.info(msg)

    # Choose batch size if not provided
    if batch_size is None:
        # Reasonable default: 2048 combos per launch; adjust as needed
        batch_size = int(cfg.get("GRID_BATCH_SIZE", 2048))

    tol_pct = float(cfg.get("RSI_TOLERANCE_PCT", 0.001))

    rows = []
    processed = 0
    start_time = time.time()

    # Batch over combos to avoid memory spikes
    for b0 in range(0, len(combos), batch_size):
        b1 = min(b0 + batch_size, len(combos))
        batch = combos[b0:b1]

        # GPU events for this batch
        streams = build_event_streams(
            df=df,
            rsi_maps=rsi_maps,
            regime_mask_host=regime_mask_host,
            combos=batch,
            backend=backend,
        )
        events_type = streams["events_type"]  # shape (B, T)

        # Price each combo’s events on host; collect summary row
        for i, combo in enumerate(batch):
            ev = events_type[i]
            sim_res = simulate_once_from_events(
                df=df,
                rsi_maps=rsi_maps,
                events_type_1d=ev,
                combo=combo,
                tolerance_pct=tol_pct,
                cfg=cfg,
                regime_label=regime_label,
                out_dir="logs",  # per-combo trades CSVs not required here; simulate_once can skip file if you prefer
                write_trades=False,  # make sure simulate_once_from_events supports skipping CSV
            )
            rows.append({
                "Buy RSI Period":   int(combo[0]),
                "Sell RSI Period":  int(combo[1]),
                "Buy RSI <":        int(combo[2]),
                "Sell RSI >":       int(combo[3]),
                "Swing Final Portfolio": float(sim_res["final_value"]),
                "Swing ROI %":      float(sim_res["roi_pct"]),
                "Swing CAGR %":     float(sim_res["cagr_pct"]),
                "Buy Count":        int(sim_res["buy_cnt"]),
                "Sell Count":       int(sim_res["sell_cnt"]),
                "Total Invested":   float(sim_res["invested_cash"]),
            })

        processed += len(batch)
        if processed % 1000 == 0 or b1 == len(combos):
            elapsed = time.time() - start_time
            msgp = f"[PROGRESS] processed {processed} / {len(combos)} combos in {elapsed:.1f}s"
            print(msgp); logger.info(msgp)

    # Verify shape: we expect exactly all bt<st combos
    if processed != len(combos):
        msg = f"[ERROR] processed {processed} != combos {len(combos)}"
        print(msg); logger.error(msg)
        raise RuntimeError(msg)

    # Build DataFrame, sort, and write per-regime CSV
    df_out = pd.DataFrame(rows)
    df_out.sort_values(by=["Swing CAGR %", "Swing ROI %"], ascending=[False, False], inplace=True)

    safe_label = _safe_label(regime_label)
    out_path = Path(out_dir) / f"summary_{safe_label}.csv"
    df_out.to_csv(out_path, index=False)

    # Final diagnostics
    msg_ok = f"[PROCESSED] combos={processed}, wrote {out_path}"
    print(msg_ok); logger.info(msg_ok)

    return str(out_path)
# --- end replace ---









# === Phase 11: Transaction Footprint (single pattern) =========================
from typing import Optional
from datetime import datetime

def _norm_regime_for_filename_exact(label: Optional[str]) -> str:
    # Keep user-visible format (e.g., gap_(7,19)), or use gap_all
    if label in (None, "gap_all", "gap_(None,None)"):
        return "gap_all"
    return str(label)

def emit_trades_csv(
    df: pd.DataFrame,
    rsi_maps: Dict[str, Dict[int, str]],
    regime_label: Optional[str],
    buy_p: int,
    sell_p: int,
    buy_thr: float,
    sell_thr: float,
    cfg: Dict[str, Any],
    out_dir: str = "logs",
) -> Dict[str, Any]:
    """
    Generate one CSV with executed transactions only, for exactly one pattern.
    CSV schema (exact order):
      Date, Regime, Action, Price, Shares, Buy RSI Period, Buy RSI Threshold,
      Sell RSI Period, Sell RSI Threshold, Cash Balance, Portfolio Balance
    Shares use integer sizing via floor(cash/price). Float64 everywhere.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Guard for single-pattern semantics (caller controls inputs; we defend here too)
    if not (isinstance(buy_p, int) and isinstance(sell_p, int)):
        raise ValueError("Transaction footprint supports exactly one pattern (single buy/sell periods).")
    if not (np.isscalar(buy_thr) and np.isscalar(sell_thr)):
        raise ValueError("Transaction footprint supports exactly one pattern (single buy/sell thresholds).")

    # Build a regime mask for the chosen label
    try:
        from src import regimes as _reg
        if regime_label in (None, "gap_all", "gap_(None,None)"):
            regime_mask_host = np.ones(len(df), dtype=bool)
        else:
            regime_mask_host = _reg.regime_mask(df, regime_label)
    except Exception:
        # Fallback: no regime cut
        regime_mask_host = np.ones(len(df), dtype=bool)

    # Build event stream for this single combo using Phase-11 kernel
    from src.gpu_events import build_event_streams, BUY, SELL
    combo = (int(buy_p), int(sell_p), float(buy_thr), float(sell_thr))
    streams = build_event_streams(
        df=df,
        rsi_maps=rsi_maps,
        regime_mask_host=regime_mask_host,
        combos=[combo],
        backend="auto",
    )
    events_type = streams["events_type"][0]     # shape (T,)
    T = len(events_type)

    # Portfolio config
    initial_cash = float(cfg.get("INITIAL_CASH", 1000.0))
    daily_cash   = float(cfg.get("DAILY_CASH", 0.0))
    tol_pct      = float(cfg.get("RSI_TOLERANCE_PCT", 0.001))

    # File name
    symbol = _infer_symbol_from_cfg(cfg) if "CSV_CACHE_FILE" in cfg else "SYMBOL"
    reg_name = _norm_regime_for_filename_exact(regime_label)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(
        out_dir,
        f"trades_{symbol}_{reg_name}_bp{buy_p}_sp{sell_p}_bt{int(buy_thr)}_st{int(sell_thr)}_{ts}.csv"
    )

    # State
    cash = initial_cash
    shares = 0.0
    close_last = float(df["Close"].iloc[-1])

    buys = sells = 0
    fast_buy = fast_sell = 0
    bsolve_buy = bsolve_sell = 0
    no_solution = 0

    # Writer
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Date","Regime","Action","Price","Shares",
            "Buy RSI Period","Buy RSI Threshold","Sell RSI Period","Sell RSI Threshold",
            "Cash Balance","Portfolio Balance"
        ])

        for d in range(T):
            # Inject daily cash at start of day
            cash += daily_cash

            evt = int(events_type[d])
            if evt not in (BUY, SELL):
                continue
            if not regime_mask_host[d]:
                # Should not happen (kernel already masked), but keep strict
                continue

            side = evt
            period = buy_p if side == BUY else sell_p
            thr    = buy_thr if side == BUY else sell_thr

            # Price resolution (FAST paths or BSOLVE) using the same helper as summary
            price, mode, _extras = _price_for_event(
                df=df, rsi_maps=rsi_maps, day_idx=d, side=side, period=period, thr=thr, tolerance_pct=tol_pct
            )
            if price is None or not np.isfinite(price) or price <= 0.0:
                no_solution += 1
                # Skip this action (log to phase log in caller if desired)
                continue

            # Execute with integer shares: all-in on BUY, all-out on SELL
            if side == BUY:
                qty = math.floor(cash / price)
                if qty <= 0:
                    # Insufficient cash to buy a single share
                    no_solution += 1
                    continue
                cost = qty * price
                cash -= cost
                shares += float(qty)
                buys += 1
                if mode == "FAST_HIGH":
                    fast_buy += 1
                elif mode == "BSOLVE":
                    bsolve_buy += 1
            else:  # SELL
                qty = shares
                if qty <= 0:
                    # No position to liquidate
                    no_solution += 1
                    continue
                proceeds = qty * price
                cash += proceeds
                shares = 0.0
                sells += 1
                if mode == "FAST_LOW":
                    fast_sell += 1
                elif mode == "BSOLVE":
                    bsolve_sell += 1

            # After action, mark the portfolio on Close[d]
            mark = float(df["Close"].iloc[d])
            portfolio = cash + shares * mark

            # Emit one row
            w.writerow([
                str(df.index[d]),
                str(df["REGIME"].iloc[d]) if "REGIME" in df.columns else (regime_label or "gap_all"),
                ("BUY" if side == BUY else "SELL"),
                f"{price:.6f}",
                f"{qty:.6f}",
                buy_p, float(buy_thr), sell_p, float(sell_thr),
                f"{cash:.6f}",
                f"{portfolio:.6f}",
            ])

    # Final accounting for banner + reconciliation
    final_value = float(cash + shares * close_last)
    invested = float(initial_cash + daily_cash * T)
    roi_pct, cagr_pct = _compute_roi_cagr(final_value, invested, df.index[0], df.index[-1])

    # Reconcile with file tail
    try:
        tail = None
        with open(path, "r", newline="") as f:
            rdr = csv.reader(f)
            next(rdr, None)  # header
            for row in rdr:
                tail = row
        if tail is not None:
            last_cash = float(tail[-2])
            last_port = float(tail[-1])
            reconcile_ok = (
                abs(last_port - final_value) < 1e-6 and
                abs(last_cash - cash) < 1e-6
            )
        else:
            # No trades: portfolio = invested if always cash; mark on last close
            reconcile_ok = True
    except Exception:
        reconcile_ok = False

    return {
        "path": path,
        "counts": {
            "buys": buys, "sells": sells,
            "fast_buy": fast_buy, "fast_sell": fast_sell,
            "bsolve_buy": bsolve_buy, "bsolve_sell": bsolve_sell,
            "no_solution": no_solution,
        },
        "final": {
            "portfolio": final_value,
            "roi_pct": roi_pct,
            "cagr_pct": cagr_pct,
            "invested": invested,
            "reconcile_ok": bool(reconcile_ok),
        },
        "pattern": {
            "buy_p": buy_p, "sell_p": sell_p, "buy_thr": float(buy_thr), "sell_thr": float(sell_thr),
            "regime": regime_label or "gap_all",
        },
    }
