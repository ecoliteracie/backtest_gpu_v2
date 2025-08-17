# src/banners/phase1.py
from __future__ import annotations
from typing import Dict, Any
from .common import ARROW

def build_banner(cfg: Dict[str, Any]) -> str:
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
        f"Date Window     : {cfg['START_DATE']} {ARROW} {cfg['END_DATE']}",
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
