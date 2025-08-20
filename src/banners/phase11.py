# src/banners/phase11.py
from __future__ import annotations
from typing import Dict

def build_banner(meta: Dict[str, object]) -> str:
    grid = meta.get("grid", {})
    gpu  = meta.get("gpu", {})
    pair = meta.get("pair", {})
    price= meta.get("price", {})
    out  = meta.get("out", {})

    lines = []
    lines.append("=" * 52)
    lines.append("Phase 11 — GPU Event Pairing + CPU Pricing")
    lines.append("=" * 52)
    lines.append(f"[GRID ] combos={grid.get('C')}, regime={grid.get('regime')}, days={grid.get('T')}")
    lines.append(f"[GPU  ] device={gpu.get('name','Unknown')}, cc={gpu.get('cc','?')}")
    lines.append(f"[PAIR ] paired_trades={pair.get('paired',0)}   overflow_combos={pair.get('overflow',0)}")
    ss = price.get("solver_stats", {})
    lines.append(
        "[PRICE] solver calls: "
        f"buy(BSOLVE={ss.get('bsolve_buy',0)}, FAST={ss.get('fast_buy',0)}), "
        f"sell(BSOLVE={ss.get('bsolve_sell',0)}, FAST={ss.get('fast_sell',0)}), "
        f"no_solution={ss.get('no_solution',0)}"
    )
    lines.append(f"[RESULT] wrote {out.get('trades_csv','(none)')}")
    lines.append("[LOG]    wrote logs/phase11_events.log")
    return "\n".join(lines)
