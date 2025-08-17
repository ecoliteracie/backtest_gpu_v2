# src/io_csv.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

# Phase 2 utilities (unchanged)
def resolve_csv_path(cfg: dict, base_dir: str | None = None) -> str:
    csv_value = str(cfg.get("CSV_CACHE_FILE", "")).strip()
    if not csv_value:
        raise ValueError("CSV_CACHE_FILE is empty")

    p = Path(os.path.expandvars(os.path.expanduser(csv_value)))
    if p.is_absolute():
        return str(p.resolve())

    if base_dir:
        root = Path(base_dir)
    else:
        root = Path(sys.argv[0]).resolve().parent

    cand1 = (root / p).resolve()
    if cand1.exists():
        return str(cand1)

    cand2 = (root / "data" / p.name).resolve()
    if cand2.exists():
        return str(cand2)

    return str(cand1)

def _fast_line_count(path: Path) -> int:
    count = 0
    with path.open("rb", buffering=1024 * 1024) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    try:
        if path.stat().st_size > 0:
            with path.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    count += 1
    except OSError:
        pass
    return count

def _read_header_and_samples(path: Path, sample_rows: int) -> tuple[str, List[str], str]:
    encodings = ["utf-8", "utf-8-sig"]
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                header = f.readline()
                samples: List[str] = []
                for _ in range(sample_rows):
                    line = f.readline()
                    if not line:
                        break
                    samples.append(line.rstrip("\r\n"))
            header = header.lstrip("\ufeff").rstrip("\r\n")
            return header, samples, enc
        except UnicodeDecodeError:
            pass

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        header = f.readline()
        samples: List[str] = []
        for _ in range(sample_rows):
            line = f.readline()
            if not line:
                break
            samples.append(line.rstrip("\r\n"))
    header = header.lstrip("\ufeff").rstrip("\r\n")
    return header, samples, "unknown"

def probe_csv(csv_path: str, sample_rows: int = 5) -> Dict[str, object]:
    p = Path(csv_path).resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"CSV not found: {p.as_posix()}")

    size_bytes = p.stat().st_size
    line_count = _fast_line_count(p)
    header, samples, encoding = _read_header_and_samples(p, sample_rows)

    return {
        "abs_path": p.as_posix(),
        "size_bytes": int(size_bytes),
        "line_count": int(line_count),
        "encoding": encoding,
        "header": header,
        "header_columns": header.split(",") if header else [],
        "samples": samples,
    }

# Phase 3: pandas DataFrame loader
def load_prices(csv_path: str):
    """
    Load CSV into a pandas DataFrame with:
      - Date parsed to datetime and set as sorted, unique index
      - Open/High/Low/Close coerced to float64
    Raises ValueError for duplicate or non-monotonic dates.
    """
    import pandas as pd

    p = Path(csv_path).resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"CSV not found: {p.as_posix()}")

    df = pd.read_csv(
        p,
        parse_dates=["Date"],
        infer_datetime_format=False,
        dtype=None,  # we will coerce core columns after read
    )

    if "Date" not in df.columns:
        raise ValueError("Missing required column: Date")

    # Coerce OHLC to float64 if present
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Indexing
    df = df.set_index("Date", drop=True)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    if not df.index.is_unique:
        dup_count = (~df.index.to_series().drop_duplicates().index.isin(df.index)).sum()
        # Simpler message:
        raise ValueError("Duplicate dates detected in index; deduplicate or aggregate before proceeding")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Date index is not monotonic increasing after sort")

    return df
