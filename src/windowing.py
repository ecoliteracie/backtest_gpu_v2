# src/windowing.py
from __future__ import annotations

import pandas as pd


def parse_iso_date(s: str) -> "pd.Timestamp":
    """
    Parse an ISO-like YYYY-MM-DD string to a timezone-naive pd.Timestamp.
    Raises ValueError with a clear message if parsing fails.
    """
    if not isinstance(s, str):
        raise ValueError(f"Invalid date value (not a string): {s!r}")
    try:
        # strict format to avoid ambiguous parsing
        ts = pd.to_datetime(s, format="%Y-%m-%d", errors="raise")
    except Exception as e:
        raise ValueError(f"Invalid ISO date string: {s!r} (expected YYYY-MM-DD)") from e
    return pd.Timestamp(ts)  # ensure Timestamp


def compute_requested_window(cfg: dict) -> tuple["pd.Timestamp", "pd.Timestamp", int]:
    """
    Read START_DATE, END_DATE, BUFFER_DAYS from cfg and return parsed values.
    """
    start_s = cfg.get("START_DATE")
    end_s = cfg.get("END_DATE")
    buffer_days = int(cfg.get("BUFFER_DAYS", 30))

    start_ts = parse_iso_date(start_s)
    end_ts = parse_iso_date(end_s)

    return start_ts, end_ts, buffer_days


def trim_for_backtest(
    df: "pd.DataFrame",
    start_ts: "pd.Timestamp",
    end_ts: "pd.Timestamp",
    buffer_days: int,
    max_period: int,
) -> tuple["pd.DataFrame", "pd.Timestamp", bool]:
    """
    Compute effective window and return the trimmed frame.

    Inclusive behavior:
    - effective_start = start_ts - buffer_days
    - df_trim = df.loc[effective_start : end_ts]  # label-based slice is inclusive on both ends
    - Warm-up rows counted as df.loc[effective_start : start_ts].shape[0] (inclusive of both ends)
      so a 30-day buffer with daily rows shows 31 (buffer + start day).

    Returns:
      df_trim, effective_start, buffer_ok
    """
    effective_start = start_ts - pd.Timedelta(days=buffer_days)

    # Trim (inclusive slice)
    df_trim = df.loc[effective_start:end_ts]

    # Warm-up sufficiency
    warmup_rows = df.loc[effective_start:start_ts].shape[0]
    buffer_ok = warmup_rows >= int(max_period)

    return df_trim, effective_start, buffer_ok
