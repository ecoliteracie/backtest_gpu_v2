# src/benchmarks.py
from __future__ import annotations

import math
import pandas as pd


def buy_and_hold(
    df_trim: "pd.DataFrame",
    initial_cash: float,
) -> dict:
    """
    Buys 1.0 *portfolio* at first valid Close in trimmed window, holds to last date.
    Does not model dividends/fees. If df_trim has <2 rows, raise ValueError.
    Returns:
      {
        "start_date": Timestamp,
        "end_date": Timestamp,
        "start_price": float,
        "end_price": float,
        "final_value": float,
        "roi_pct": float,
        "cagr_pct": float,
        "days_held": int
      }
    """
    if df_trim.shape[0] < 2:
        raise ValueError("Buy-and-hold requires at least 2 rows in trimmed window.")
    if "Close" not in df_trim.columns:
        raise ValueError("Missing required column: Close")

    start_date = df_trim.index[0]
    end_date   = df_trim.index[-1]

    start_px = float(df_trim["Close"].iloc[0])
    end_px   = float(df_trim["Close"].iloc[-1])

    if math.isnan(start_px) or math.isnan(end_px):
        raise ValueError("Close has NaN at boundary dates (start or end).")

    if initial_cash <= 0:
        raise ValueError("initial_cash must be > 0")

    # Fully invest at start
    shares = initial_cash / start_px
    final_value = shares * end_px

    roi_pct = (final_value / initial_cash - 1.0) * 100.0

    # CAGR based on calendar days in window
    days = (end_date - start_date).days
    if days <= 0:
        cagr_pct = 0.0
    else:
        years = days / 365.25
        cagr_pct = ((final_value / initial_cash) ** (1.0 / years) - 1.0) * 100.0

    return {
        "start_date": start_date, "end_date": end_date,
        "start_price": start_px, "end_price": end_px,
        "final_value": final_value, "roi_pct": roi_pct, "cagr_pct": cagr_pct,
        "days_held": days
    }
