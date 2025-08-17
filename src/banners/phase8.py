# src/banners/phase8.py
from __future__ import annotations
from typing import Dict, List

def _fmt_samples(kind: str, dates: List[str]) -> str:
    if not dates:
        return f"{kind} idx: (none)"
    return f"{kind} idx: " + ", ".join(dates)

def build_banner(meta: Dict[str, object]) -> str:
    params = meta.get("params", {})
    counts = meta.get("counts", {})
    samples = meta.get("first_true_dates", {})
    regime = meta.get("regime", {})
    backend = meta.get("backend", {})

    lines = []
    lines.append("=" * 53)
    lines.append("Phase 8 — GPU Buy/Sell Predicate Masks")
    lines.append("=" * 53)
    lines.append(f"[PARAMS] buy_period={params.get('buy_period')}, sell_period={params.get('sell_period')}, "
                 f"buy_thr={params.get('buy_thr')}, sell_thr={params.get('sell_thr')}")
    lines.append(f"[COUNTS] buy_ok={counts.get('buy_ok',0)}, sell_ok={counts.get('sell_ok',0)}")

    buy_line = _fmt_samples("buy", samples.get("buy", []))
    sell_line = _fmt_samples("sell", samples.get("sell", []))
    lines.append(f"[SAMPLES] {buy_line} | {sell_line}")

    applied = regime.get("applied")
    true_cnt = regime.get("true_count", 0)
    lines.append(f"[REGIME] applied={applied if applied is not None else 'None'}, true_count={true_cnt}")

    dev_name = backend.get("name", "Unknown")
    cc = backend.get("cc", "?")
    lines.append(f"[BACKEND] CuPy device={dev_name}, cc={cc}")
    lines.append("[LOG]     wrote logs/phase08_masks_gpu.log")
    return "\n".join(lines)
