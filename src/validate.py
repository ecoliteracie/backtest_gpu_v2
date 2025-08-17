# src/validate.py
from __future__ import annotations

import re
from typing import List, Dict

def require_columns(df, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

def summarize_indicators(df) -> Dict[str, object]:
    # RSI close-only periods (RSI_2_CLOSE, RSI_3_CLOSE, ...)
    rsi_pat = re.compile(r"^RSI_(\d+)_CLOSE$")
    rsi_periods = []
    for col in df.columns:
        m = rsi_pat.match(col)
        if m:
            rsi_periods.append(int(m.group(1)))
    rsi_periods = sorted(set(rsi_periods))

    has_ma50 = "MA_50" in df.columns
    has_ma200 = "MA_200" in df.columns

    # Short list of extra indicators (informative, not exhaustive)
    ignore = set(["Open", "High", "Low", "Close"])
    ignore.update([f"RSI_{p}_CLOSE" for p in rsi_periods])
    if has_ma50:
        ignore.add("MA_50")
    if has_ma200:
        ignore.add("MA_200")

    extras = [c for c in df.columns if c not in ignore]
    # Keep a concise preview
    extras_preview = extras[:8]

    return {
        "rsi_close_periods": rsi_periods,
        "has_ma50": has_ma50,
        "has_ma200": has_ma200,
        "extras_preview": extras_preview,
    }
