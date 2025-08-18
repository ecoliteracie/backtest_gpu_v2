# src/banners/phase10.py
from __future__ import annotations
from typing import Dict, List

def _fmt_samples(kind: str, dates: List[str]) -> str:
    if not dates:
        return f"{kind} idx: (none)"
    return f"{kind} idx: " + ", ".join(dates)

def _fmt_boundary_line(boundary_assignments: Dict[str, Dict[str, int]]) -> str:
    if not boundary_assignments:
        return "[CHECK] boundary assignments : (none)"
    parts: List[str] = []
    for b, dests in boundary_assignments.items():
        sub = ", ".join([f"to {lab}={cnt}" for lab, cnt in dests.items()])
        parts.append(f"{b}: {sub}")
    return "[CHECK] boundary assignments : " + " | ".join(parts)

def build_banner(meta: Dict[str, object]) -> str:
    params   = meta.get("params", {})
    totals   = meta.get("totals", {})
    backend  = meta.get("backend", {})
    perreg   = meta.get("per_regime", [])
    checks   = meta.get("checks", {})

    lines: List[str] = []
    lines.append("=" * 52)
    lines.append("Phase 10 — Masks by Regime (GPU)")
    lines.append("=" * 52)
    lines.append(f"[PARAMS] buy_period={params.get('buy_period')}, sell_period={params.get('sell_period')}, "
                 f"buy_thr={params.get('buy_thr')}, sell_thr={params.get('sell_thr')}")
    lines.append(f"[TOTALS] buy_ok={totals.get('buy_all',0)}, sell_ok={totals.get('sell_all',0)}")
    dev_name = backend.get("name", "Unknown")
    cc = backend.get("cc", "?")
    lines.append(f"[BACKEND] CuPy device={dev_name}, cc={cc}")

    for block in perreg:
        lab = block.get("label", "unknown")
        cnt = block.get("counts", {})
        smp = block.get("samples", {})
        lines.append(f"[REGIME] {lab}")
        lines.append(f"[COUNTS] buy_ok={cnt.get('buy_ok',0)}, sell_ok={cnt.get('sell_ok',0)}")
        buy_line  = _fmt_samples("buy",  smp.get("buy", []))
        sell_line = _fmt_samples("sell", smp.get("sell", []))
        lines.append(f"[SAMPLES] {buy_line} | {sell_line}")

    # Domain reconciliation & integrity checks
    lines.append(f"[CHECK] totals_all_rows      : buy={totals.get('buy_all',0)}, sell={totals.get('sell_all',0)}")
    lines.append(f"[CHECK] totals_in_ma_gap_dom : buy={totals.get('buy_dom',0)}, sell={totals.get('sell_dom',0)}")

    sum_buy = checks.get("sum_buy", 0)
    sum_sell = checks.get("sum_sell", 0)
    ok_buy = (sum_buy == totals.get('buy_dom',0))
    ok_sell = (sum_sell == totals.get('sell_dom',0))
    lines.append(f"[CHECK] sum_per_regime==dom? : buy={ok_buy} (sum={sum_buy} vs dom={totals.get('buy_dom',0)}), "
                 f"sell={ok_sell} (sum={sum_sell} vs dom={totals.get('sell_dom',0)})")

    lines.append(f"[CHECK] in_NaN_domain        : buy={totals.get('buy_nan',0)}, sell={totals.get('sell_nan',0)}")

    overlaps = checks.get("overlaps", {"count": 0, "first_date": None})
    holes    = checks.get("holes",    {"count": 0, "first_date": None})
    lines.append(f"[CHECK] overlaps             : {overlaps.get('count',0)}, first_date={overlaps.get('first_date')}")
    lines.append(f"[CHECK] holes                : {holes.get('count',0)}, first_date={holes.get('first_date')}")

    boundary_assignments = checks.get("boundary_assignments", {})
    lines.append(_fmt_boundary_line(boundary_assignments))

    lines.append("[LOG]    wrote logs/phase10_masks_by_regime.log")
    return "\n".join(lines)
