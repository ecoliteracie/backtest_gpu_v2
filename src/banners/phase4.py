# src/banners/phase4.py
from __future__ import annotations

def build_banner(
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
    rows_before_start_total = df.loc[:requested_start].shape[0]
    rows_before_start_trim = df_trim.loc[:requested_start].shape[0]
    rows_dropped_pre = rows_before_start_total - rows_before_start_trim
    rows_dropped_post = rows_total - rows_trim - rows_dropped_pre

    data_first = df.index.min()
    data_last = df.index.max()
    trim_first = df_trim.index.min() if rows_trim else None
    trim_last = df_trim.index.max() if rows_trim else None

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
