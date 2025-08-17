# src/banners/phase7.py
from __future__ import annotations
from typing import Dict, List

def _fmt_period_list(periods: List[int]) -> str:
    if not periods:
        return "(none)"
    if len(periods) == 1:
        return f"{periods[0]}"
    return f"{periods[0]}..{periods[-1]} (count={len(periods)})"

def _fmt_variant_presence(name: str, mp: Dict[int, str]) -> str:
    if not mp:
        return f"[{name}]    none"
    ps = ", ".join(str(p) for p in sorted(mp.keys()))
    return f"[{name}]    present for: {ps}"

def _fmt_missing(missing: Dict[int, List[str]]) -> str:
    if not missing:
        return "[MISSING] none"
    parts = []
    for p in sorted(missing.keys()):
        parts.append(f"period {p}: {','.join(missing[p])}")
    return "[MISSING] " + "; ".join(parts)

def _fmt_nan_line(nans: Dict[str, Dict[str, int]], cols_in_order: List[str]) -> str:
    parts = []
    picked = []
    for key in cols_in_order:
        if key in nans:
            picked.append(key)
        if len(picked) >= 6:
            break
    if not picked:
        return "[NANS]    none"
    for k in picked:
        parts.append(f"{k}: head={nans[k]['head']}, total={nans[k]['total']}")
    return "[NANS]    " + "; ".join(parts)

def _fmt_ordering(diag: Dict[str, object], periods: List[int]) -> List[str]:
    lines = []
    lines.append(f"[ORDER]   sample_size={diag['ordering']['sample_per_period']} per period (where available)")
    viols = diag["ordering"]["violations"]
    if not viols:
        lines.append("violations: none")
        return lines
    parts = []
    for p in periods:
        if p in viols:
            parts.append(f"p={p}:{viols[p]['count']}")
    if parts:
        lines.append("violations: " + ", ".join(parts))
    for p in periods:
        if p in viols and viols[p]['count'] > 0 and viols[p]['dates']:
            lines.append(f"  p={p} sample dates: " + ", ".join(viols[p]['dates']))
    return lines

def build_banner(rsi_maps: Dict[str, object], diag: Dict[str, object]) -> str:
    periods = rsi_maps["periods"]
    close_map = rsi_maps["close_map"]
    low_map   = rsi_maps.get("low_map", {})
    high_map  = rsi_maps.get("high_map", {})
    open_map  = rsi_maps.get("open_map", {})
    missing   = rsi_maps.get("missing_variants", {})

    lines = []
    lines.append("=" * 52)
    lines.append("Phase 7 — RSI Column Binding & Invariants")
    lines.append("=" * 52)
    lines.append(f"[PERIODS] detected={_fmt_period_list(periods)}")
    if close_map:
        items = [f"{p}:{close_map[p]}" for p in periods]
        lines.append("[CLOSE]   " + ", ".join(items))
    else:
        lines.append("[CLOSE]   none")
    lines.append(_fmt_variant_presence("LOW", low_map))
    lines.append(_fmt_variant_presence("HIGH", high_map))
    lines.append(_fmt_variant_presence("OPEN", open_map))
    lines.append(_fmt_missing(missing))

    cols_priority = []
    if 2 in close_map:  cols_priority.append(close_map[2])
    if 14 in close_map: cols_priority.append(close_map[14])
    for p in periods:
        c = close_map[p]
        if c not in cols_priority:
            cols_priority.append(c)
    lines.append(_fmt_nan_line(diag["nans"], cols_priority))

    if diag["bounds"]["ok"]:
        lines.append("[BOUNDS]  All detected RSI columns within [0,100] beyond warm-up")
    else:
        fv = diag["bounds"]["first_violation"]
        if fv:
            lines.append(f"[BOUNDS]  FAIL at {fv['index']} in {fv['column']}: {fv['value']}")
        else:
            lines.append("[BOUNDS]  FAIL (first violation not captured)")

    lines.extend(_fmt_ordering(diag, periods))
    lines.append("[LOG]     wrote logs/phase07_rsi_columns.log")
    return "\n".join(lines)
