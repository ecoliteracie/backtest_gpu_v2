# src/banners/phase2.py
from __future__ import annotations

def build_banner(probe: dict) -> str:
    size_mb = probe["size_bytes"] / (1024 * 1024) if probe["size_bytes"] else 0.0
    head = probe.get("header", "")
    samples = probe.get("samples", [])

    lines = []
    lines.append("=" * 53)
    lines.append("Phase 2 — CSV Presence & Header Probe")
    lines.append("=" * 53)
    lines.append("")
    lines.append(f"[PATH]  {probe['abs_path']}")
    lines.append(f"[FILE]  exists=Yes, size={size_mb:.1f} MB, lines≈{probe['line_count']}")
    if head:
        lines.append(f"[HEAD]  columns={head}")
    else:
        lines.append("[HEAD]  columns=<empty or unreadable>")
    for i, s in enumerate(samples, 1):
        lines.append(f"[SAMPLE {i}] {s}")
    lines.append(f"[LOG]   Wrote: logs/phase02_csv_probe.log")
    return "\n".join(lines)
