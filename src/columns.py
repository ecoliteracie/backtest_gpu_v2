# src/columns.py
from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


_PERIODS = list(range(2, 15))


def _normalize(name: str) -> str:
    """
    Normalize a column name for robust matching:
    - strip spaces
    - uppercase
    """
    return re.sub(r"\s+", "", name).upper()


def _find_close_col(cols_norm_map: Dict[str, str], p: int) -> Optional[str]:
    """
    CLOSE priority:
      1) RSI_{p}
      2) RSI_{p}_CLOSE
    """
    cand1 = f"RSI_{p}"
    cand2 = f"RSI_{p}_CLOSE"
    if cand1 in cols_norm_map:
        return cols_norm_map[cand1]
    if cand2 in cols_norm_map:
        return cols_norm_map[cand2]
    return None


def _find_variant_col(cols_norm_map: Dict[str, str], p: int, variant: str) -> Optional[str]:
    """
    Accept flexible patterns for LOW/HIGH/OPEN like:
      RSI*{p}*LOW (stars mean 'anything, incl. underscores')
    The match is:
      - starts with 'RSI'
      - contains exact period {p} (digit-bounded)
      - ends with the VARIANT token
    """
    variant = variant.upper()
    pat = re.compile(rf"^RSI.*(?<!\d){p}(?!\d).*{variant}$", re.IGNORECASE)
    for key_norm, orig in cols_norm_map.items():
        if pat.match(key_norm):
            return orig
    return None


def detect_rsi_columns(df: pd.DataFrame) -> Dict[str, object]:
    """
    Discover available RSI columns for periods in [2..14].

    Returns:
      {
        "periods": List[int],
        "close_map": Dict[int, str],
        "low_map":   Dict[int, str],
        "high_map":  Dict[int, str],
        "open_map":  Dict[int, str],
        "missing_variants": Dict[int, List[str]],
      }

    Notes:
      - A period is considered available iff a CLOSE variant was found.
      - When both RSI_{p} and RSI_{p}_CLOSE exist, prefer RSI_{p}.
    """
    # Build normalization map: normalized -> original
    cols_norm_map: Dict[str, str] = {_normalize(c): c for c in df.columns}

    close_map: Dict[int, str] = {}
    low_map: Dict[int, str] = {}
    high_map: Dict[int, str] = {}
    open_map: Dict[int, str] = {}

    for p in _PERIODS:
        close_col = _find_close_col(cols_norm_map, p)
        if close_col is None:
            continue
        close_map[p] = close_col

        # Optional variants
        low_col  = _find_variant_col(cols_norm_map, p, "LOW")
        high_col = _find_variant_col(cols_norm_map, p, "HIGH")
        open_col = _find_variant_col(cols_norm_map, p, "OPEN")
        if low_col:  low_map[p] = low_col
        if high_col: high_map[p] = high_col
        if open_col: open_map[p] = open_col

    periods = sorted(close_map.keys())
    if not periods:
        raise ValueError("No CLOSE-based RSI columns detected for periods 2..14.")

    # Missing variants accounting
    missing_variants: Dict[int, List[str]] = {}
    for p in periods:
        miss = []
        if p not in low_map:  miss.append("LOW")
        if p not in high_map: miss.append("HIGH")
        if p not in open_map: miss.append("OPEN")
        if miss:
            missing_variants[p] = miss

    return {
        "periods": periods,
        "close_map": close_map,
        "low_map": low_map,
        "high_map": high_map,
        "open_map": open_map,
        "missing_variants": missing_variants,
    }


def analyze_rsi_invariants(df: pd.DataFrame, rsi_maps: Dict[str, object], rng_seed: int = 7) -> Dict[str, object]:
    """
    Compute diagnostics required by Phase 7:
      - presence counts (by variant)
      - bounds check (hard): values in [0, 100] beyond warm-up
      - NaN accounting (total + head NaN run-length)
      - ordering expectation sample:
          check RSI_LOW <= RSI_OPEN <= RSI_CLOSE <= RSI_HIGH when
          Low <= Open <= Close <= High and all four RSI variants exist.
      - dtype info (float64 warning)
    """
    rng = np.random.default_rng(rng_seed)

    close_map = rsi_maps["close_map"]
    low_map   = rsi_maps.get("low_map", {})
    high_map  = rsi_maps.get("high_map", {})
    open_map  = rsi_maps.get("open_map", {})
    periods   = rsi_maps["periods"]

    # Presence
    presence = {
        "count_periods": len(periods),
        "variants": {
            "CLOSE": len(close_map),
            "LOW":   len(low_map),
            "HIGH":  len(high_map),
            "OPEN":  len(open_map),
        },
    }

    # NaN accounting & dtype warnings; also prepare warm-up indices
    nan_stats: Dict[str, Dict[str, int]] = {}
    non_float64: List[str] = []
    first_non_nan_idx: Dict[str, int] = {}

    def head_nan_runlength(s: pd.Series) -> int:
        # number of leading NaNs from top until first non-NaN
        isna = s.isna().to_numpy()
        if not isna.any():
            return 0
        nz = np.flatnonzero(~isna)
        return int(nz[0]) if nz.size > 0 else len(s)

    all_cols: List[str] = []
    for m in (close_map, low_map, high_map, open_map):
        all_cols.extend(list(m.values()))
    seen = set()
    for col in all_cols:
        if col in seen:
            continue
        seen.add(col)

        s = df[col]
        if s.dtype != "float64":
            non_float64.append(col)
        head = head_nan_runlength(s)
        total = int(s.isna().sum())
        nan_stats[col] = {"head": head, "total": total}
        first_non_nan_idx[col] = head  # integer position

    # Bounds (hard): Must be within [0, 100] after warm-up
    bounds_ok = True
    bounds_reason = None
    first_violation = None

    def check_bounds(col: str) -> Optional[Tuple[object, float]]:
        s = df[col]
        start = first_non_nan_idx.get(col, 0)
        v = s.iloc[start:]
        mask = v.notna() & ((v < 0.0) | (v > 100.0))
        if mask.any():
            first_pos = mask.to_numpy().argmax()
            idx_first = v.index[first_pos]
            val_first = float(v[mask].iloc[0])
            return (idx_first, val_first)
        return None

    for col in seen:
        res = check_bounds(col)
        if res is not None:
            bounds_ok = False
            idx_first, val_first = res
            bounds_reason = f"Out-of-range value in column '{col}' at index {idx_first!r}: {val_first} (expected within [0,100])."
            first_violation = {"column": col, "index": idx_first, "value": val_first}
            break

    # Ordering expectation sampling
    ordering = {"sample_per_period": 200, "violations": {}}
    if all(c in df.columns for c in ["Low", "Open", "Close", "High"]):
        ohlc_ok = (df["Low"] <= df["Open"]) & (df["Open"] <= df["Close"]) & (df["Close"] <= df["High"])
        idx_ok = np.flatnonzero(ohlc_ok.to_numpy())
    else:
        idx_ok = np.array([], dtype=int)

    for p in periods:
        if not (p in low_map and p in open_map and p in high_map):
            continue  # need all four variants to test ordering
        cols = (low_map[p], open_map[p], close_map[p], high_map[p])
        ok_mask = df[list(cols)].notna().all(axis=1).to_numpy()
        idx_rsi = np.flatnonzero(ok_mask)

        if idx_ok.size > 0:
            idx_candidates = np.intersect1d(idx_ok, idx_rsi, assume_unique=False)
        else:
            idx_candidates = idx_rsi

        if idx_candidates.size == 0:
            ordering["violations"][p] = {"count": 0, "dates": []}
            continue

        sample_size = min(200, idx_candidates.size)
        sample_idx = rng.choice(idx_candidates, size=sample_size, replace=False)

        v_low  = df.iloc[sample_idx][cols[0]].to_numpy()
        v_open = df.iloc[sample_idx][cols[1]].to_numpy()
        v_close= df.iloc[sample_idx][cols[2]].to_numpy()
        v_high = df.iloc[sample_idx][cols[3]].to_numpy()
        viol_mask = ~((v_low <= v_open) & (v_open <= v_close) & (v_close <= v_high))
        viol_count = int(viol_mask.sum())
        viol_dates = []
        if viol_count > 0:
            viol_idx = sample_idx[np.flatnonzero(viol_mask)]
            for i in viol_idx[:5]:
                viol_dates.append(str(df.index[i]))
        ordering["violations"][p] = {"count": viol_count, "dates": viol_dates}

    return {
        "presence": presence,
        "bounds": {"ok": bounds_ok, "reason": bounds_reason, "first_violation": first_violation},
        "nans": nan_stats,
        "ordering": ordering,
        "dtypes": {"non_float64": non_float64},
    }
