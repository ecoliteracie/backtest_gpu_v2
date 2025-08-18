# src/regimes.py
from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np

import pandas as pd


def compute_ma_gap(df: pd.DataFrame) -> pd.Series:
    """
    Ensure MA_50 / MA_200 exist (compute if missing), then compute:
        MA_GAP = ((MA_50 - MA_200) / max(MA_50, MA_200)) * 100
    Row-wise, float64. Avoid inf when denominator is 0 by setting NaN.
    Attaches df["MA_GAP"] and returns the Series.
    """
    # Ensure float64 Close
    if "Close" not in df.columns:
        raise ValueError("Missing 'Close' column required for MA computation.")
    if df["Close"].dtype != "float64":
        df["Close"] = df["Close"].astype("float64")

    # Ensure MA_50
    if "MA_50" not in df.columns:
        df["MA_50"] = (
            df["Close"].rolling(window=50, min_periods=50).mean().astype("float64")
        )
    else:
        if df["MA_50"].dtype != "float64":
            df["MA_50"] = df["MA_50"].astype("float64")

    # Ensure MA_200
    if "MA_200" not in df.columns:
        df["MA_200"] = (
            df["Close"].rolling(window=200, min_periods=200).mean().astype("float64")
        )
    else:
        if df["MA_200"].dtype != "float64":
            df["MA_200"] = df["MA_200"].astype("float64")

    ma50 = df["MA_50"]
    ma200 = df["MA_200"]

    # Row-wise denominator (Series aligned to index)
    den = pd.Series(
        np.maximum(ma50.to_numpy(), ma200.to_numpy()),
        index=df.index,
        dtype="float64",
    )

    # Prepare result as NaN
    gap = pd.Series(np.nan, index=df.index, dtype="float64")

    # Valid where both are finite and denominator != 0
    valid = ma50.notna() & ma200.notna() & (den != 0.0)
    gap.loc[valid] = ((ma50.loc[valid] - ma200.loc[valid]) / den.loc[valid]) * 100.0

    # Attach
    df["MA_GAP"] = gap.astype("float64")
    return df["MA_GAP"]


def generate_regime_labels(gap_ranges: List[Tuple[float | None, float | None]]) -> List[str]:
    """
    For each (low, high) produce a canonical label like:
        gap_(None,-19), gap_(-19,7), ...
    """
    labels: List[str] = []
    for low, high in gap_ranges:
        labels.append(f"gap_({low},{high})")
    return labels


def label_by_ranges(
    df: pd.DataFrame,
    gap_ranges: List[Tuple[float | None, float | None]],
) -> pd.Series:
    """
    Using df["MA_GAP"], assign each row to exactly one bucket by first-match.
    A row belongs to (low, high) if:
      (low is None or MA_GAP >= low) and (high is None or MA_GAP < high)
    Attaches df["REGIME"] and returns the Series.
    """
    if "MA_GAP" not in df.columns:
        raise ValueError("MA_GAP not present. Call compute_ma_gap(df) first.")

    ma_gap = df["MA_GAP"]
    labels = generate_regime_labels(gap_ranges)

    out = pd.Series(pd.NA, index=df.index, dtype="object")
    assigned = pd.Series(False, index=df.index, dtype="bool")

    for (low, high), label in zip(gap_ranges, labels):
        cond_low = True if low is None else (ma_gap >= low)
        cond_high = True if high is None else (ma_gap < high)
        mask = (~ma_gap.isna()) & (~assigned) & cond_low & cond_high
        if mask.any():
            out.loc[mask] = label
            assigned.loc[mask] = True

    # Attach if empty or identical; else overwrite (safe)
    df["REGIME"] = out
    return df["REGIME"]


def regime_mask(df: pd.DataFrame, regime_label: str) -> np.ndarray:
    """
    Return a host boolean mask aligned to df.index:
      - "gap_all" -> all True
      - otherwise df["REGIME"] == regime_label
    """
    n = len(df.index)
    if regime_label == "gap_all":
        return np.ones(n, dtype=bool)
    if "REGIME" not in df.columns:
        raise ValueError("REGIME not present. Call label_by_ranges(df, ...) first.")
    return (df["REGIME"] == regime_label).to_numpy(dtype=bool, copy=False)
