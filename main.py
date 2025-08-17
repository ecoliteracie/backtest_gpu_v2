# main.py — Phase 1 + Phase 2 + Phase 3 + Phase 4
from __future__ import annotations

import sys
from pathlib import Path

from src.logging_setup import get_logger
from src.config_loader import load, validate
from src.io_csv import resolve_csv_path, probe_csv, load_prices
from src.validate import require_columns, summarize_indicators
from src.windowing import compute_requested_window, trim_for_backtest


def _ensure_dirs() -> None:
    for d in ("logs", "results"):
        Path(d).mkdir(parents=True, exist_ok=True)


def _banner_phase1(cfg: dict) -> str:
    rsi = cfg["RSI_PERIODS"]
    buy = cfg["BUY_THRESHOLDS"]
    sell = cfg["SELL_THRESHOLDS"]
    gaps = cfg["GAP_RANGES"]

    def rng(values):
        return f"{min(values):g}..{max(values):g}"

    def fmt_gap(g):
        lo, hi = g
        lo_s = "-inf" if lo is None else f"{lo:g}"
        hi_s = "inf" if hi is None else f"{hi:g}"
        return f"({lo_s}, {hi_s})"

    gap_examples = ", ".join(fmt_gap(g) for g in gaps[:2])
    lines = [
        "# Phase 1 — Bootstrap & Config Load",
        "",
        "[CONFIG] OK",
        f"Date Window     : {cfg['START_DATE']} \u2192 {cfg['END_DATE']}",
        f"CSV Cache File  : {cfg['CSV_CACHE_FILE']}",
        f"Cash            : INITIAL={cfg['INITIAL_CASH']:.2f}, DAILY={float(cfg['DAILY_CASH']):.2f}",
        f"RSI Periods     : count={len(rsi)}, range={min(rsi)}..{max(rsi)}",
        (
            "Thresholds      : "
            f"BUY count={len(buy)}, range={rng(buy)}; "
            f"SELL count={len(sell)}, range={rng(sell)}"
        ),
        f"GAP Ranges      : count={len(gaps)}, e.g. [{gap_examples}{', ...' if len(gaps) > 2 else ''}]",
        f"Buffer Days     : {cfg['BUFFER_DAYS']}",
        f"RSI Tolerance % : {cfg['RSI_TOLERANCE_PCT']}",
        f"[LOG] Wrote: logs/phase01_init.log",
    ]
    return "\n".join(lines)


def _banner_phase2(probe: dict) -> str:
    size_mb = probe["size_bytes"] / (1024 * 1024) if probe["size_bytes"] else 0.0
    head = probe.get("header", "")
    samples = probe.get("samples", [])
    lines = []
    lines.append("=" * 53)
    lines.append("Phase 2 — CSV Presence & Header Probe")
    lines.append("=" * 53)
    lines.append("")
    lines.append(f"[PATH]  {probe['abs_path']}")
    lines.append(f"[FILE]  exists=Yes, size={size_mb:.1f} MB, lines≈{probe['line_count']}")
    if head:
        lines.append(f"[HEAD]  columns={head}")
    else:
        lines.append("[HEAD]  columns=<empty or unreadable>")
    for i, s in enumerate(samples, 1):
        lines.append(f"[SAMPLE {i}] {s}")
    lines.append(f"[LOG]   Wrote: logs/phase02_csv_probe.log")
    return "\n".join(lines)


def _banner_phase3(df) -> str:
    import pandas as pd

    rows, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    dmin = df.index.min()
    dmax = df.index.max()

    nulls = {c: int(df[c].isna().sum()) for c in ["Open", "High", "Low", "Close"] if c in df.columns}

    metrics = summarize_indicators(df)
    rsi_periods = metrics["rsi_close_periods"]
    rsi_summary = "none"
    if rsi_periods:
        rsi_summary = f"{min(rsi_periods)}..{max(rsi_periods)} (count={len(rsi_periods)})"

    ma_flags = f"MA_50={'Yes' if metrics['has_ma50'] else 'No'}, MA_200={'Yes' if metrics['has_ma200'] else 'No'}"

    def fmt_row(idx, row):
        vals = []
        for c in ["Open", "High", "Low", "Close"]:
            if c in row.index:
                v = row[c]
                vals.append(f"{v:.6f}" if v == v else "NaN")
        return f"{idx.date()} " + "  ".join(vals)

    head_lines = [fmt_row(idx, row) for idx, row in df.head(3).iterrows()]
    tail_lines = [fmt_row(idx, row) for idx, row in df.tail(3).iterrows()]

    lines = []
    lines.append("# Phase 3 — DataFrame Load & Column Sanity")
    lines.append("")
    lines.append(f"[DATA] rows={rows}, cols={cols}, memory={mem_mb:.1f} MB")
    lines.append(f"[DATE] range={dmin.date()} \u2192 {dmax.date()}")
    lines.append("[COLUMNS] Date index ok; required: Open, High, Low, Close present")
    lines.append(f"[NULLS] Open={nulls.get('Open', 0)}, High={nulls.get('High', 0)}, Low={nulls.get('Low', 0)}, Close={nulls.get('Close', 0)}")
    lines.append("[INDICATORS]")
    lines.append(f"RSI_CLOSE periods: {rsi_summary}")
    lines.append(f"MA flags: {ma_flags}")
    if metrics["extras_preview"]:
        lines.append("Extras: " + ", ".join(metrics["extras_preview"]) + (" ..." if len(metrics["extras_preview"]) >= 8 else ""))
    lines.append("[HEAD 3]")
    lines.extend(head_lines)
    lines.append("[TAIL 3]")
    lines.extend(tail_lines)
    lines.append(f"[LOG] Wrote: logs/phase03_df_load.log")
    return "\n".join(lines)


def _banner_phase4(
    df,
    df_trim,
    requested_start,
    requested_end,
    effective_start,
    buffer_ok: bool,
    max_period: int,
) -> str:
    rows_total = df.shape[0]
    rows_trim = df_trim.shape[0]
    # before start date
    rows_before_start_total = df.loc[:requested_start].shape[0]
    rows_before_start_trim = df_trim.loc[:requested_start].shape[0]
    rows_dropped_pre = rows_before_start_total - rows_before_start_trim
    rows_dropped_post = rows_total - rows_trim - rows_dropped_pre

    data_first = df.index.min()
    data_last = df.index.max()
    trim_first = df_trim.index.min() if rows_trim else None
    trim_last = df_trim.index.max() if rows_trim else None

    # Warm-up count (inclusive both ends; mirrors trim_for_backtest logic)
    warmup_rows = df.loc[effective_start:requested_start].shape[0]

    lines = []
    lines.append("=" * 53)
    lines.append("Phase 4 — Date Window Trim & Buffer Check")
    lines.append("=" * 53)
    lines.append(f"[REQUEST] start={requested_start.date()}, end={requested_end.date()}, buffer_days={int((requested_start - effective_start).days)}, max_rsi_period={max_period}")
    lines.append(f"[EFFECTIVE RANGE] {effective_start.date()} \u2192 {requested_end.date()}")
    lines.append(f"[DATA RANGE]      {data_first.date()} \u2192 {data_last.date()}")
    if trim_first is None or trim_last is None:
        lines.append("[TRIMMED RANGE]   <no rows>")
    else:
        lines.append(f"[TRIMMED RANGE]   {trim_first.date()} \u2192 {trim_last.date()}")
    lines.append(f"[ROWS] total={rows_total}, trimmed={rows_trim}, dropped_pre={rows_dropped_pre}, dropped_post={rows_dropped_post}")
    lines.append(f"[BUFFER] warmup_rows_between({effective_start.date()}, {requested_start.date()})={warmup_rows}, required={max_period}, ok={str(buffer_ok)}")
    lines.append("[LOG] Wrote: logs/phase04_window.log")
    return "\n".join(lines)


def main() -> int:
    _ensure_dirs()

    # Phase 1
    log1 = Path("logs") / "phase01_init.log"
    logger1 = get_logger("phase01", log1)
    try:
        cfg = load("configs.default")
    except Exception as e:
        msg = f"CONFIG ERROR: cannot import configs.default ({e})"
        print(msg)
        logger1.error(msg)
        return 1
    try:
        cfg = validate(cfg)
    except Exception as e:
        msg = f"CONFIG ERROR: {e}"
        print(msg)
        logger1.error(msg)
        return 1
    banner1 = _banner_phase1(cfg)
    print(banner1)
    for line in banner1.splitlines():
        logger1.info(line)

    # Phase 2
    log2 = Path("logs") / "phase02_csv_probe.log"
    logger2 = get_logger("phase02", log2)
    try:
        csv_abs = resolve_csv_path(cfg, base_dir=str(Path(__file__).resolve().parent))
        probe = probe_csv(csv_abs, sample_rows=5)
    except FileNotFoundError as e:
        msg = str(e)
        print(msg)
        logger2.error(msg)
        return 1
    except Exception as e:
        msg = f"CSV probe failed: {e}"
        print(msg)
        logger2.error(msg)
        return 1
    banner2 = _banner_phase2(probe)
    print()
    print(banner2)
    for line in banner2.splitlines():
        logger2.info(line)

    # Phase 3
    log3 = Path("logs") / "phase03_df_load.log"
    logger3 = get_logger("phase03", log3)
    try:
        df = load_prices(csv_abs)
        require_columns(df, ["Open", "High", "Low", "Close"])
    except Exception as e:
        msg = f"DataFrame load/validate failed: {e}"
        print(msg)
        logger3.error(msg)
        return 1
    banner3 = _banner_phase3(df)
    print()
    print(banner3)
    for line in banner3.splitlines():
        logger3.info(line)

    # Phase 4
    log4 = Path("logs") / "phase04_window.log"
    logger4 = get_logger("phase04", log4)
    try:
        start_ts, end_ts, buffer_days = compute_requested_window(cfg)
        max_period = max(cfg["RSI_PERIODS"])
        df_trim, effective_start, buffer_ok = trim_for_backtest(
            df, start_ts, end_ts, buffer_days, max_period
        )
        if df_trim.shape[0] == 0:
            msg = "No rows in trimmed window; check your date range and buffer."
            print(msg)
            logger4.error(msg)
            return 1
    except Exception as e:
        msg = f"Windowing failed: {e}"
        print(msg)
        logger4.error(msg)
        return 1

    banner4 = _banner_phase4(
        df=df,
        df_trim=df_trim,
        requested_start=start_ts,
        requested_end=end_ts,
        effective_start=effective_start,
        buffer_ok=buffer_ok,
        max_period=max_period,
    )
    print()
    print(banner4)
    for line in banner4.splitlines():
        logger4.info(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
