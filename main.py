# src/main.py
# Phase 3 — Date window + lookback buffer + Buy-and-Hold baseline
# Single runner with minimal CLI. Keeps Phase 2 summary and adds window slicing + BNH.

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from types import ModuleType
import importlib
import importlib.util
from typing import Tuple, Dict, Any, List

import pandas as pd


# ---------------------------
# Config loading
# ---------------------------
def _load_config_module() -> Tuple[ModuleType, str]:
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
    for attr in ("DATA_FILES", "CSV_FILES", "CSV_MAP", "CSV_BY_SYMBOL", "SYMBOL_TO_CSV"):
        mapping = getattr(cfg, attr, None)
        if isinstance(mapping, dict) and symbol in mapping:
            return str(mapping[symbol])

    csv_cache = getattr(cfg, "CSV_CACHE_FILE", None)
    if isinstance(csv_cache, (str, Path)) and csv_cache:
        return Path(csv_cache).name

    return f"{symbol}_full_ohlc_indicators.csv"


def _resolve_csv_path(filename: str | Path) -> Path:
    filename = Path(filename).name
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
    return (project_root / "data" / filename)


# ---------------------------
# Config value helpers
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
    return "SOXL"


def _symbol_settings(cfg: ModuleType, symbol: str) -> Dict[str, Any]:
    # Prefer SETTINGS[symbol]
    settings = getattr(cfg, "SETTINGS", {})
    if isinstance(settings, dict) and symbol in settings and isinstance(settings[symbol], dict):
        return dict(settings[symbol])

    # Otherwise look for top-level defaults
    top = {}
    for key in (
        "START_DATE",
        "END_DATE",
        "INITIAL_CASH",
        "DAILY_CASH",
        "EXPENSE_RATIO",
        "DISTRIBUTION_YIELD",
    ):
        if hasattr(cfg, key):
            top[key] = getattr(cfg, key)
    return top


def _get_with_fallback(s: Dict[str, Any], k: str, default: Any) -> Any:
    v = s.get(k, None)
    return default if v is None else v


# ---------------------------
# CLI
# ---------------------------
def _parse_args(cfg: ModuleType) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--symbol", type=str, default=_default_symbol(cfg))
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--buffer-days", type=int, default=None)
    parser.add_argument("--write-bnh", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser.parse_args()


# ---------------------------
# Phase 2 core pieces
# ---------------------------
def _load_csv(path: Path) -> Tuple[pd.DataFrame, bool, bool]:
    df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    required = ["Date", "Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        print(f"[ERROR] Missing required columns: {', '.join(missing)}")
        sys.exit(1)

    # Numeric coercion (safe)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pre_dates = df["Date"]
    pre_unsorted = not pre_dates.is_monotonic_increasing
    pre_dupes = not pre_dates.is_unique

    # Stable sort + reindex
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)

    if df.shape[0] == 0:
        print("[ERROR] Empty dataset after load")
        sys.exit(1)

    return df, pre_unsorted, pre_dupes


def _print_phase2_summary(df: pd.DataFrame, file_print_path: str, symbol: str, head_n: int, quiet: bool,
                          pre_unsorted: bool, pre_dupes: bool) -> None:
    print("=== Phase 3: Window Slice + Buy-and-Hold Baseline ===")
    print(f"[FILE] symbol={symbol}")
    print(f"[FILE] path={file_print_path}")

    rows, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[DATA] rows={rows}, cols={cols}")
    print(f"[DATA] memory={mem_mb:.1f} MB")

    first_s = df.loc[0, "Date"].strftime("%Y-%m-%d")
    last_s = df.loc[rows - 1, "Date"].strftime("%Y-%m-%d")
    print(f"[DATE] range={first_s} \u2192 {last_s}")

    cols_list = list(map(str, df.columns.tolist()))
    display_cols = (cols_list[:10] + ["..."]) if len(cols_list) > 10 else cols_list
    print(f"[COLUMNS] {', '.join(display_cols)}")

    if pre_unsorted or pre_dupes:
        print("[WARN] Duplicate or unsorted dates detected before sort; data was re-sorted")

    required = ["Date", "Open", "High", "Low", "Close"]
    null_counts = df[required].isna().sum()
    print("[NULLS] " + ", ".join(f"{c}={int(null_counts[c])}" for c in required))

    if not quiet:
        if head_n < 0:
            head_n = 0
        subset = df[required]
        if head_n > 0:
            print(f"[HEAD {head_n}]")
            print(subset.head(head_n).to_string(index=False))
            print(f"[TAIL {head_n}]")
            print(subset.tail(head_n).to_string(index=False))
        else:
            print(f"[HEAD {head_n}]")
            print(f"[TAIL {head_n}]")


# ---------------------------
# Phase 3 helpers
# ---------------------------
def _slice_with_buffer(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, buffer_days: int
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_buf = start_date - pd.Timedelta(days=int(buffer_days))
    df_buf = df[(df["Date"] >= start_buf) & (df["Date"] <= end_date)].copy()
    df_trim = df_buf[(df_buf["Date"] >= start_date) & (df_buf["Date"] <= end_date)].copy()
    return df_buf.reset_index(drop=True), df_trim.reset_index(drop=True)


def _range_str(df: pd.DataFrame) -> str:
    if df.empty:
        return "NA \u2192 NA"
    return f"{df['Date'].iloc[0].strftime('%Y-%m-%d')} \u2192 {df['Date'].iloc[-1].strftime('%Y-%m-%d')}"


def _compute_bnh(df_trim: pd.DataFrame, initial_cash: float, daily_cash: float,
                 expense_ratio: float, distribution_yield: float
                 ) -> Tuple[Dict[str, Any], pd.DataFrame]:
    # Drop NaN closes inside trimmed window
    before = len(df_trim)
    df_bnh = df_trim.dropna(subset=["Close"]).copy()
    dropped = before - len(df_bnh)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows with NaN Close during baseline")
    if len(df_bnh) < 2:
        print("[ERROR] Insufficient rows after trim (need ≥ 2)")
        sys.exit(1)

    cash = float(initial_cash)
    shares = 0.0
    invested = float(initial_cash)

    def yq(ts: pd.Timestamp) -> Tuple[int, int]:
        q = (int(ts.month) - 1) // 3 + 1
        return int(ts.year), q

    prev_yq = None
    records: List[Dict[str, Any]] = []

    for i, row in df_bnh.iterrows():
        d: pd.Timestamp = row["Date"]
        close: float = float(row["Close"])
        curr_yq = yq(d)

        # Quarter-change: apply prior quarter at the first row of new quarter using previous day's close
        if prev_yq is not None and curr_yq != prev_yq and i > 0:
            prev_close = float(df_bnh.loc[i - 1, "Close"])
            shares *= (1.0 - expense_ratio / 4.0)
            cash += shares * prev_close * (distribution_yield / 4.0)

        prev_yq = curr_yq

        # Daily cash injection and integer shares buy at Close
        cash += float(daily_cash)
        invested += float(daily_cash)
        units = int(cash // close)
        if units > 0:
            cash -= units * close
            shares += float(units)

        port = cash + shares * close
        records.append(
            {
                "Date": d,
                "Close": close,
                "Cash": cash,
                "Shares": shares,
                "Portfolio_Value": port,
            }
        )

    # Final-quarter adjustment once at series end using last day's close
    last_close = float(df_bnh["Close"].iloc[-1])
    shares *= (1.0 - expense_ratio / 4.0)
    cash += shares * last_close * (distribution_yield / 4.0)

    # Update last record to reflect end-of-quarter adjustment
    records[-1]["Cash"] = cash
    records[-1]["Shares"] = shares
    records[-1]["Portfolio_Value"] = cash + shares * last_close

    ts = pd.DataFrame.from_records(records)
    years = (df_bnh["Date"].iloc[-1] - df_bnh["Date"].iloc[0]).days / 365.25
    years = float(years) if years > 0 else 0.0

    final_value = float(ts["Portfolio_Value"].iloc[-1])
    roi_pct = 100.0 * (final_value - invested) / invested if invested > 0 else 0.0
    ann_pct = 100.0 * ((final_value / invested) ** (1.0 / years) - 1.0) if years > 0 and invested > 0 else 0.0

    out = {
        "start": df_bnh["Date"].iloc[0].strftime("%Y-%m-%d"),
        "end": df_bnh["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "years": years,
        "final_value": final_value,
        "invested": invested,
        "roi_pct": roi_pct,
        "ann_pct": ann_pct,
    }
    return out, ts


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    cfg, _cfg_path_display = _load_config_module()
    args = _parse_args(cfg)

    symbol = args.symbol.upper().strip()
    head_n = max(int(args.head), 0)

    filename = _csv_filename_for_symbol(cfg, symbol)
    csv_path = _resolve_csv_path(filename)
    file_print_path = f"data/{Path(filename).name}"

    # Load full CSV and print Phase 2 summary (unchanged)
    df, pre_unsorted, pre_dupes = _load_csv(csv_path)
    _print_phase2_summary(df, file_print_path, symbol, head_n, args.quiet, pre_unsorted, pre_dupes)

    # Resolve dates, buffer, and per-symbol params (CLI overrides take precedence)
    sym_cfg = _symbol_settings(cfg, symbol)

    start_date_str = args.start_date if args.start_date else _get_with_fallback(sym_cfg, "START_DATE", None)
    end_date_str = args.end_date if args.end_date else _get_with_fallback(sym_cfg, "END_DATE", None)
    if not start_date_str or not end_date_str:
        print("[ERROR] START_DATE and END_DATE must be provided (via config or CLI)")
        sys.exit(1)

    try:
        start_date = pd.to_datetime(str(start_date_str)).tz_localize(None)
        end_date = pd.to_datetime(str(end_date_str)).tz_localize(None)
    except Exception:
        print("[ERROR] Invalid start or end date format (expected YYYY-MM-DD)")
        sys.exit(1)

    if end_date < start_date:
        print("[ERROR] END_DATE must be on or after START_DATE")
        sys.exit(1)

    global_buffer = getattr(cfg, "LOOKBACK_BUFFER_DAYS", 50)
    buffer_days = int(args.buffer_days) if args.buffer_days is not None else int(global_buffer)

    # Slice with buffer, then trim
    df_buf, df_trim = _slice_with_buffer(df, start_date, end_date, buffer_days)

    print(f"[SLICE] buffer_days={buffer_days}")
    print(f"[SLICE] kept={len(df_buf)}, range={_range_str(df_buf)}")

    print(f"[TRIM] kept={len(df_trim)}, range={_range_str(df_trim)}")

    if len(df_trim) < 2:
        print("[ERROR] Insufficient rows after trim (need ≥ 2)")
        sys.exit(1)

    # Buy-and-Hold baseline parameters (fallbacks to 0.0 where absent)
    initial_cash = float(_get_with_fallback(sym_cfg, "INITIAL_CASH", 0.0))
    daily_cash = float(_get_with_fallback(sym_cfg, "DAILY_CASH", 0.0))
    expense_ratio = float(_get_with_fallback(sym_cfg, "EXPENSE_RATIO", 0.0))
    distribution_yield = float(_get_with_fallback(sym_cfg, "DISTRIBUTION_YIELD", 0.0))

    # Compute baseline
    bnh, ts = _compute_bnh(df_trim, initial_cash, daily_cash, expense_ratio, distribution_yield)

    # Terminal-verifiable BNH summary
    print(
        "[BNH] "
        f"initial_cash=${initial_cash:,.2f}  daily_cash=${daily_cash:,.2f}  "
        f"expense={expense_ratio*100:.2f}%  dist={distribution_yield*100:.2f}%"
    )
    print(f"[BNH] period={bnh['start']} \u2192 {bnh['end']}  years={bnh['years']:.2f}")
    print(
        "[BNH] "
        f"final_value=${bnh['final_value']:,.2f}, invested=${bnh['invested']:,.2f}, "
        f"total_return=${(bnh['final_value'] - bnh['invested']):,.2f}"
    )
    print(f"[BNH] ROI={bnh['roi_pct']:.2f}%  annualized={bnh['ann_pct']:.2f}%")

    # Optional CSV output
    if args.write_bnh:
        results_dir = Path(__file__).resolve().parents[1] / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"BNH_{symbol}_{bnh['start']}_{bnh['end']}.csv"
        out_path = results_dir / out_name
        ts_out = ts.copy()
        ts_out["Date"] = ts_out["Date"].dt.strftime("%Y-%m-%d")
        ts_out.to_csv(out_path, index=False)
        print(f"[OUT] {out_path.relative_to(results_dir.parent)}")


if __name__ == "__main__":
    main()
