# src/banners/common.py
from __future__ import annotations

ARROW = "\u2192"

def format_bytes(n: int | None) -> str:
    """Render bytes as MiB/GiB for display."""
    if n is None:
        return "n/a"
    step = 1024.0
    g = n / (step**3)
    if g >= 1.0:
        return f"{g:.1f} GiB"
    m = n / (step**2)
    return f"{m:.1f} MiB"

def fmt_money(x: float) -> str:
    return f"{x:,.2f}"

def fmt_float(x: float, places: int = 6) -> str:
    return f"{x:.{places}f}"
