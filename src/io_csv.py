# src/io_csv.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List


def resolve_csv_path(cfg: dict, base_dir: str | None = None) -> str:
    """
    Resolve CSV_CACHE_FILE to an absolute path.
    - If absolute, normalize and return.
    - If relative, resolve against base_dir (project root). If not provided,
      resolve against the directory containing main.py (sys.argv[0]).
    - If the bare filename does not exist at base_dir, also try base_dir/data/<name>.
    """
    csv_value = str(cfg.get("CSV_CACHE_FILE", "")).strip()
    if not csv_value:
        raise ValueError("CSV_CACHE_FILE is empty")

    p = Path(os.path.expandvars(os.path.expanduser(csv_value)))

    if p.is_absolute():
        return str(p.resolve())

    # Determine project root
    if base_dir:
        root = Path(base_dir)
    else:
        # Directory where main.py lives
        root = Path(sys.argv[0]).resolve().parent

    # First candidate: relative to project root
    cand1 = (root / p).resolve()
    if cand1.exists():
        return str(cand1)

    # Second candidate: <root>/data/<filename>
    cand2 = (root / "data" / p.name).resolve()
    if cand2.exists():
        return str(cand2)

    # Return normalized absolute path (first candidate) even if missing;
    # caller will raise a clear FileNotFoundError with this absolute path
    return str(cand1)


def _fast_line_count(path: Path) -> int:
    """Count lines quickly in binary mode."""
    count = 0
    with path.open("rb", buffering=1024 * 1024) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    # If file is non-empty and does not end with newline, add 1
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
    """
    Try to read header + up to sample_rows data lines.
    Encoding strategy:
      - try utf-8
      - then utf-8-sig
      - else open with utf-8(errors='replace') and mark as 'unknown'
    Returns: (header_line, sample_lines, encoding_label)
    """
    encodings = ["utf-8", "utf-8-sig"]
    last_err = None
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
            # Strip BOM if any survived
            header = header.lstrip("\ufeff").rstrip("\r\n")
            return header, samples, enc
        except UnicodeDecodeError as e:
            last_err = e

    # Fallback: unknown encoding; still attempt reading with replacement
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
    """
    Validate file exists, then collect:
      - abs_path
      - size_bytes
      - line_count
      - encoding (best effort)
      - header (raw line string)
      - header_columns (simple comma split; quotes left as-is)
      - samples (list[str], up to sample_rows)
    """
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
