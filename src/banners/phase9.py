# src/banners/phase9.py
from __future__ import annotations
from typing import List, Tuple, Dict


def _fmt_ranges(gap_ranges: List[Tuple[float | None, float | None]]) -> str:
    parts = []
    for low, high in gap_ranges:
        parts.append(f"({low},{high})")
    return ", ".join(parts)


def _fmt_counts_line(labels: List[str], counts: Dict[str, int]) -> str:
    parts = [f"{lab}={counts.get(lab,0)}" for lab in labels]
    return "[COUNTS] " + ", ".join(parts)


def _fmt_samples_block(labels: List[str], samples: Dict[str, list[str]]) -> list[str]:
    lines = ["[SAMPLES] first 3 dates per regime:"]
    for lab in labels:
        arr = samples.get(lab, [])
        if not arr:
            lines.append(f"{lab}: (none)")
        else:
            lines.append(f"{lab}: {', '.join(arr)}")
    return lines


def build_banner(
    gap_ranges: List[Tuple[float | None, float | None]],
    ma_gap_dtype: str,
    non_null: int,
    total: int,
    first_valid: str | None,
    labels: List[str],
    counts: Dict[str, int],
    samples: Dict[str, list[str]],
) -> str:
    lines: list[str] = []
    lines.append("=" * 52)
    lines.append("Phase 9 — Regime Labels & Counts")
    lines.append("=" * 52)
    lines.append(f"[RANGES] {_fmt_ranges(gap_ranges)}")
    fv = first_valid if first_valid is not None else "None"
    lines.append(f"[MA_GAP] dtype={ma_gap_dtype}, non_null={non_null} / total={total}, first_valid={fv}")
    lines.append(_fmt_counts_line(labels, counts))
    lines.extend(_fmt_samples_block(labels, samples))
    lines.append("[LOG]    wrote logs/phase09_regimes.log")
    return "\n".join(lines)
