# src/main.py
# Phase 2 — Load CSV and print verifiable dataset stats
# Notes:
#   - Single runner (this file) with minimal CLI (--symbol, --head).
#   - Imports config from configs/default.py (fallback to config/default.py).
#   - Resolves CSV under project_root/data without hard-coded absolute paths.
#   - Prints ONLY the specified blocks; exits(1) on hard validation failures.

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from types import ModuleType
import importlib
import importlib.util

import pandas as pd


# ---------------------------
# Config loading
# ---------------------------
def _load_config_module() -> tuple[ModuleType, str]:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    for mod_name in ("configs.default", "config.default"):
        try:
            mod = importlib.import_module(mod_name)
            return mod, mod_name.replace(".", "/") + ".py"
        except ModuleNotFoundError:
            continue

    # Fallback: direct path import
    for rel in (("configs", "default.py"), ("config", "default.py")):
        p = project_root.joinpath(*rel)
        if p.exists():
            spec = importlib.util.spec_from_file_location("default", p)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod, str(Path(*rel))

    raise ImportError("Could not locate configs/default.py or config/default.py")


# ---------------------------
# CSV resolution helpers
# ---------------------------
def _csv_filename_for_symbol(cfg: ModuleType, symbol: str) -> str:
    # Try common mapping names
    for attr in ("DATA_FILES", "CSV_FILES", "CSV_MAP", "CSV_BY_SYMBOL", "SYMBOL_TO_CSV"):
        mapping = getattr(cfg, attr, None)
        if isinstance(mapping, dict) and symbol in mapping:
            return str(mapping[symbol])

    # Single-file fallback from config
    csv_cache = getattr(cfg, "CSV_CACHE_FILE", None)
    if isinstance(csv_cache, (str, Path)) and csv_cache:
        return Path(csv_cache).name  # print as data/<name>

    # Infer conventional name
    return f"{symbol}_full_ohlc_indicators.csv"


def _resolve_csv_path(filename: str | Path) -> Path:
    """
    Return Path by searching common locations relative to project root and CWD.
    """
    filename = Path(filename).name  # normalize to just the name
    project_root = Path(__file__).resolve().parents[1]

    candidates = [
        project_root / "data" / filename,
        project_root / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    # If not found, still return the first expected path for error context
    return (project_root / "data" / filename)


# ---------------------------
# CLI / Defaults
# ---------------------------
def _default_symbol(cfg: ModuleType) -> str:
    syms = getattr(cfg, "SYMBOLS", None)
    if isinstance(syms, (list, tuple)) and syms:
        return str(syms[0])
    settings = getattr(cfg, "SETTINGS", {})
    if isinstance(settings, dict):
        for k, v in settings.items():
            if isinstance(v, dict) and v.get("ACTIVE", False):
                return str(k)
    # Fallback
    return "SOXL"


def _parse_args(cfg: ModuleType) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--symbol", type=str, default=_default_symbol(cfg))
    parser.add_argument("--head", type=int, default=3)
    return parser.parse_args()


# ---------------------------
# Core
# ---------------------------
def main() -> None:
    cfg, _cfg_path_display = _load_config_module()
    args = _parse_args(cfg)

    symbol = args.symbol.upper().strip()
    head_n = max(int(args.head), 0)

    filename = _csv_filename_for_symbol(cfg, symbol)
    csv_path = _resolve_csv_path(filename)

    # For printing, keep a deterministic "data/<file>" style path
    file_print_path = f"data/{Path(filename).name}"

    # Load CSV robustly
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["Date"],
            low_memory=False,
        )
    except FileNotFoundError:
        # Keep console minimal; one-line error then exit.
        print(f"[ERROR] CSV not found: {file_print_path}")
        sys.exit(1)

    # Required columns
    required = ["Date", "Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        print(f"[ERROR] Missing required columns: {', '.join(missing)}")
        sys.exit(1)

    # Coerce numeric safely (without raising)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Detect pre-sort issues
    pre_dates = df["Date"]
    pre_unsorted = not pre_dates.is_monotonic_increasing
    pre_dupes = not pre_dates.is_unique

    # Sort and reindex
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)

    # Post-load validation
    if df.shape[0] == 0:
        print("[ERROR] Empty dataset after load")
        sys.exit(1)

    # Prints (ONLY the specified blocks)
    print("=== Phase 2: CSV Load & Inspect ===")
    print(f"[FILE] symbol={symbol}")
    print(f"[FILE] path={file_print_path}")

    rows, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[DATA] rows={rows}, cols={cols}")
    print(f"[DATA] memory={mem_mb:.1f} MB")

    # Date range (after sorting)
    first_date = df.loc[0, "Date"]
    last_date = df.loc[rows - 1, "Date"]
    # Ensure string format YYYY-MM-DD
    first_s = first_date.strftime("%Y-%m-%d")
    last_s = last_date.strftime("%Y-%m-%d")
    print(f"[DATE] range={first_s} \u2192 {last_s}")

    # Columns (first 10 only)
    cols_list = list(map(str, df.columns.tolist()))
    if len(cols_list) > 10:
        display_cols = cols_list[:10] + ["..."]
    else:
        display_cols = cols_list
    print(f"[COLUMNS] {', '.join(display_cols)}")

    # Warn if duplicates or unsorted before sort
    if pre_unsorted or pre_dupes:
        print("[WARN] Duplicate or unsorted dates detected before sort; data was re-sorted.")

    # Null checks for key columns
    null_counts = df[required].isna().sum()
    print(
        "[NULLS] "
        + ", ".join(f"{c}={int(null_counts[c])}" for c in required)
    )

    # Head / Tail previews (subset)
    subset = df[required]
    if head_n > 0:
        print(f"[HEAD {head_n}]")
        print(subset.head(head_n).to_string(index=False))
        print(f"[TAIL {head_n}]")
        print(subset.tail(head_n).to_string(index=False))
    else:
        # If N=0, still print empty sections for determinism
        print(f"[HEAD {head_n}]")
        print(f"[TAIL {head_n}]")


if __name__ == "__main__":
    main()
