# main.py — Phase 1 + Phase 2
from __future__ import annotations

import sys
from pathlib import Path

from src.logging_setup import get_logger
from src.config_loader import load, validate
from src.io_csv import resolve_csv_path, probe_csv


def _ensure_dirs() -> None:
    for d in ("logs", "results"):
        Path(d).mkdir(parents=True, exist_ok=True)


def _banner_phase1(cfg: dict) -> str:
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
        f"Date Window     : {cfg['START_DATE']} \u2192 {cfg['END_DATE']}",
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


def _banner_phase2(probe: dict) -> str:
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


def main() -> int:
    _ensure_dirs()

    # Phase 1
    log1 = Path("logs") / "phase01_init.log"
    logger1 = get_logger("phase01", log1)
    try:
        cfg = load("configs.default")
    except Exception as e:
        msg = f"CONFIG ERROR: cannot import configs.default ({e})"
        print(msg)
        logger1.error(msg)
        return 1
    try:
        cfg = validate(cfg)
    except Exception as e:
        msg = f"CONFIG ERROR: {e}"
        print(msg)
        logger1.error(msg)
        return 1
    banner1 = _banner_phase1(cfg)
    print(banner1)
    # log lines individually to keep readability in the file
    for line in banner1.splitlines():
        logger1.info(line)

    # Phase 2
    log2 = Path("logs") / "phase02_csv_probe.log"
    logger2 = get_logger("phase02", log2)
    try:
        csv_abs = resolve_csv_path(cfg, base_dir=str(Path(__file__).resolve().parent))
        probe = probe_csv(csv_abs, sample_rows=5)
    except FileNotFoundError as e:
        msg = str(e)
        print(msg)
        logger2.error(msg)
        return 1
    except Exception as e:
        # Any unexpected probe error
        msg = f"CSV probe failed: {e}"
        print(msg)
        logger2.error(msg)
        return 1

    banner2 = _banner_phase2(probe)
    print()
    print(banner2)
    for line in banner2.splitlines():
        logger2.info(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
