# main.py
from __future__ import annotations
from pathlib import Path
import sys

from src.logging_setup import get_logger
from src.config_loader import load, validate



def _fmt_range(vals):
    return f"{int(min(vals))}..{int(max(vals))}"

def _fmt_gap_tuple(t):
    lo, hi = t
    def f(x):
        if x is None:
            return "-inf" if x is lo else "inf"
        # display without .0 when integer-like
        return str(int(x)) if float(x).is_integer() else str(float(x))
    return f"({f(lo)}, {f(hi)})"

def main() -> int:
    # Ensure expected folders exist
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    log_file = "logs/phase01_init.log"
    log = get_logger("phase01", log_file)

    try:
        cfg_raw = load("configs.default")
        cfg = validate(cfg_raw)
    except (ImportError, ValueError, TypeError) as e:
        print("# Phase 1 — Bootstrap & Config Load")
        print(f"[CONFIG] ERROR: {e}")
        return 1

    # Build banner lines
    start, end = cfg["START_DATE"], cfg["END_DATE"]
    csv_path = cfg["CSV_CACHE_FILE"]
    init_cash = float(cfg["INITIAL_CASH"])
    daily_cash = float(cfg["DAILY_CASH"])
    rsi_list = cfg["RSI_PERIODS"]
    buy_list = cfg["BUY_THRESHOLDS"]
    sell_list = cfg["SELL_THRESHOLDS"]
    gaps = cfg["GAP_RANGES"]
    buf_days = int(cfg["BUFFER_DAYS"])
    tol = cfg["RSI_TOLERANCE_PCT"]

    lines = [
        "# Phase 1 — Bootstrap & Config Load",
        "",
        "[CONFIG] OK",
        f"Date Window     : {start} \u2192 {end}",
        f"CSV Cache File  : {csv_path}",
        f"Cash            : INITIAL={init_cash:0.2f}, DAILY={daily_cash:0.2f}",
        f"RSI Periods     : count={len(rsi_list)}, range={_fmt_range(rsi_list)}",
        (
            "Thresholds      : "
            f"BUY count={len(buy_list)}, range={_fmt_range(buy_list)}; "
            f"SELL count={len(sell_list)}, range={_fmt_range(sell_list)}"
        ),
        (
            "GAP Ranges      : "
            f"count={len(gaps)}, e.g. [{_fmt_gap_tuple(gaps[0])}"
            + (f", {_fmt_gap_tuple(gaps[1])}" if len(gaps) > 1 else "")
            + ", ...]"
        ),
        f"Buffer Days     : {buf_days}",
        f"RSI Tolerance % : {tol}",
        f"[LOG] Wrote: {log_file}",
    ]

    # Emit to both console and file via the logger
    for line in lines:
        log.info(line)

    return 0

if __name__ == "__main__":
    sys.exit(main())
