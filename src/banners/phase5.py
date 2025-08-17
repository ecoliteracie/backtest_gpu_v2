# src/banners/phase5.py
from __future__ import annotations
from .common import fmt_money

def build_banner(bh: dict, initial_cash: float) -> str:
    lines = []
    lines.append("# Phase 5 — Buy-and-Hold Baseline")
    lines.append("")
    lines.append(f"[WINDOW]  start={bh['start_date'].date()} \u2192 end={bh['end_date'].date()}, days={bh['days_held']}")
    lines.append(f"[PRICES]  start_close={bh['start_price']:.6f}, end_close={bh['end_price']:.6f}")
    lines.append(f"[CASH]    initial={fmt_money(initial_cash)} \u2192 final={fmt_money(bh['final_value'])}")
    lines.append(f"[METRICS] ROI={bh['roi_pct']:.3f}%, CAGR={bh['cagr_pct']:.3f}%")
    lines.append("[LOG]     Wrote: logs/phase05_bh.log")
    return "\n".join(lines)
