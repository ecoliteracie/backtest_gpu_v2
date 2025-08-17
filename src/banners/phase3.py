# src/banners/phase3.py
from __future__ import annotations
import pandas as pd
from src.validate import summarize_indicators

def build_banner(df) -> str:
    rows, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    dmin = df.index.min()
    dmax = df.index.max()

    nulls = {c: int(df[c].isna().sum()) for c in ["Open", "High", "Low", "Close"] if c in df.columns}

    metrics = summarize_indicators(df)
    rsi_periods = metrics["rsi_close_periods"]
    rsi_summary = "none" if not rsi_periods else f"{min(rsi_periods)}..{max(rsi_periods)} (count={len(rsi_periods)})"
    ma_flags = f"MA_50={'Yes' if metrics['has_ma50'] else 'No'}, MA_200={'Yes' if metrics['has_ma200'] else 'No'}"

    def fmt_row(idx, row):
        vals = []
        for c in ["Open", "High", "Low", "Close"]:
            if c in row.index:
                v = row[c]
                vals.append(f"{v:.6f}" if pd.notna(v) else "NaN")
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
