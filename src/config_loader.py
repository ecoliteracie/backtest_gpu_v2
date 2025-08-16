# src/config_loader.py
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, List

UPPER_SNAKE = lambda s: s.isupper() and all(c.isalnum() or c == "_" for c in s)

def _import_by_module_path(module_path: str):
    return importlib.import_module(module_path)

def _import_by_filesystem_fallback(module_path: str):
    caller_dir = Path.cwd()
    candidates = [
        caller_dir / "configs" / "default.py",
        caller_dir / "config" / "default.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("configs.default", p)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
                return module
    raise ImportError("cannot locate configs/default.py via filesystem fallback")

def load(config_module_path: str = "configs.default", symbol: str | None = None) -> Dict[str, Any]:
    """
    Import configs/default.py and return a flat dict of required keys.
    Supports:
      A) True top-level constants (START_DATE, END_DATE, ...)
      B) SETTINGS dict with per-symbol entries (SOXL/TQQQ/etc.)
    """
    # Import module (dotted path first, then filesystem fallback)
    try:
        mod = _import_by_module_path(config_module_path)
    except Exception:
        mod = _import_by_filesystem_fallback(config_module_path)

    # 1) Consider real top-level constants, explicitly excluding SETTINGS
    required_keys = {
        "START_DATE","END_DATE","GAP_RANGES","RSI_PERIODS",
        "BUY_THRESHOLDS","SELL_THRESHOLDS","CSV_CACHE_FILE","INITIAL_CASH"
    }
    top_level: Dict[str, Any] = {}
    for name, value in inspect.getmembers(mod):
        if name == "SETTINGS":
            continue  # do not treat SETTINGS as a constant config
        if UPPER_SNAKE(name) and not name.startswith("__"):
            top_level[name] = value

    # If all required keys are present as true constants, use them
    if required_keys.issubset(top_level.keys()):
        return top_level

    # 2) Otherwise, parse the SETTINGS dict (per-symbol layout)
    if hasattr(mod, "SETTINGS") and isinstance(mod.SETTINGS, dict):
        return _from_settings(mod.SETTINGS, symbol)

    # 3) Nothing usable found
    raise ValueError(
        "No valid top-level constants and no usable SETTINGS dict found in configs/default.py"
    )


def _from_settings(settings: dict, symbol: str | None) -> Dict[str, Any]:
    # Choose symbol entry
    chosen_key: str | None = None
    chosen: dict | None = None

    if symbol and symbol in settings and isinstance(settings[symbol], dict):
        chosen_key = symbol
        chosen = settings[symbol]
    else:
        for k, v in settings.items():
            if isinstance(v, dict) and v.get("ACTIVE") is True:
                chosen_key = k
                chosen = v
                break

    if not chosen:
        raise ValueError("SETTINGS has no ACTIVE=True symbol (and no explicit symbol provided)")

    # Required direct mappings
    def need(key: str):
        if key not in chosen:
            raise ValueError(f"SETTINGS[{chosen_key}] missing required key: {key}")
        return chosen[key]

    cfg: Dict[str, Any] = {}
    cfg["START_DATE"] = need("START_DATE")
    cfg["END_DATE"] = need("END_DATE")
    cfg["CSV_CACHE_FILE"] = need("CSV_CACHE_FILE")
    cfg["INITIAL_CASH"] = need("INITIAL_CASH")
    cfg["DAILY_CASH"] = chosen.get("DAILY_CASH", 0.0)

    # GAP_RANGES
    gap_ranges = chosen.get("GAP_RANGES") or settings.get("GAP_RANGES")
    if not gap_ranges:
        raise ValueError(f"SETTINGS[{chosen_key}] missing GAP_RANGES")
    cfg["GAP_RANGES"] = gap_ranges

    # RSI_PERIODS (prefer explicit list; else build from RSI_MIN..RSI_MAX; else use RSI_WINDOW as a single value)
    if "RSI_PERIODS" in chosen:
        cfg["RSI_PERIODS"] = chosen["RSI_PERIODS"]
    else:
        rmin = chosen.get("RSI_MIN")
        rmax = chosen.get("RSI_MAX")
        rwin = chosen.get("RSI_WINDOW")
        if rmin is not None and rmax is not None:
            cfg["RSI_PERIODS"] = list(range(int(rmin), int(rmax) + 1))
        elif rwin is not None:
            cfg["RSI_PERIODS"] = [int(rwin)]
        else:
            raise ValueError(f"SETTINGS[{chosen_key}] must define RSI_PERIODS or RSI_MIN/RSI_MAX or RSI_WINDOW")

    # BUY/SELL thresholds (prefer explicit lists; else create minimal lists from *_MIN/_MAX)
    if "BUY_THRESHOLDS" in chosen:
        cfg["BUY_THRESHOLDS"] = chosen["BUY_THRESHOLDS"]
    else:
        bmin = chosen.get("BUY_THRESHOLD_MIN")
        bmax = chosen.get("BUY_THRESHOLD_MAX", bmin)
        if bmin is None:
            raise ValueError(f"SETTINGS[{chosen_key}] missing BUY_THRESHOLD_MIN")
        cfg["BUY_THRESHOLDS"] = [float(bmin)] if (bmax is None or bmax == bmin) else [float(bmin), float(bmax)]

    if "SELL_THRESHOLDS" in chosen:
        cfg["SELL_THRESHOLDS"] = chosen["SELL_THRESHOLDS"]
    else:
        smin = chosen.get("SELL_THRESHOLD_MIN")
        smax = chosen.get("SELL_THRESHOLD_MAX", smin)
        if smin is None:
            raise ValueError(f"SETTINGS[{chosen_key}] missing SELL_THRESHOLD_MIN")
        cfg["SELL_THRESHOLDS"] = [float(smin)] if (smax is None or smax == smin) else [float(smin), float(smax)]

    # Optional defaults and aliases
    cfg["BUFFER_DAYS"] = settings.get("LOOKBACK_BUFFER_DAYS", 30)
    cfg["RSI_TOLERANCE_PCT"] = chosen.get("RSI_TOLERANCE_PCT", chosen.get("RSI_PROXIMITY_PCT", 0.00001))

    # Phase note
    cfg["_NOTE"] = "Date ordering not validated in Phase 1"
    return cfg

def _ensure_iterable_ints(x: Any, key: str) -> List[int]:
    if isinstance(x, range):
        return list(x)
    if isinstance(x, (list, tuple, set)):
        try:
            return [int(v) for v in x]
        except Exception:
            raise ValueError(f"{key}: expected iterable of ints")
    raise ValueError(f"{key}: expected iterable of ints")

def _ensure_iterable_numbers(x: Any, key: str) -> List[float]:
    if isinstance(x, range):
        return [float(v) for v in x]
    if isinstance(x, (list, tuple, set)):
        try:
            return [float(v) for v in x]
        except Exception:
            raise ValueError(f"{key}: expected iterable of numbers")
    raise ValueError(f"{key}: expected iterable of numbers")

def _validate_gap_ranges(x: Any) -> List[Tuple[float | None, float | None]]:
    if not isinstance(x, (list, tuple)):
        raise ValueError("GAP_RANGES: expected list of (low, high) tuples")
    out: List[Tuple[float | None, float | None]] = []
    for i, item in enumerate(x):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError(f"GAP_RANGES[{i}]: expected a 2-tuple (low, high)")
        low, high = item
        if low is not None and not isinstance(low, (int, float)):
            raise ValueError(f"GAP_RANGES[{i}].low: expected float or None")
        if high is not None and not isinstance(high, (int, float)):
            raise ValueError(f"GAP_RANGES[{i}].high: expected float or None")
        if low is not None and high is not None and not (low < high):
            raise ValueError(f"GAP_RANGES[{i}]: low must be < high when both present")
        out.append((float(low) if low is not None else None,
                    float(high) if high is not None else None))
    if not out:
        raise ValueError("GAP_RANGES: list must not be empty")
    return out

def validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = [
        "START_DATE",
        "END_DATE",
        "GAP_RANGES",
        "RSI_PERIODS",
        "BUY_THRESHOLDS",
        "SELL_THRESHOLDS",
        "CSV_CACHE_FILE",
        "INITIAL_CASH",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"missing required keys: {', '.join(missing)}")

    if not isinstance(cfg["START_DATE"], str):
        raise ValueError("START_DATE: expected str (ISO-like)")
    if not isinstance(cfg["END_DATE"], str):
        raise ValueError("END_DATE: expected str (ISO-like)")

    cfg["GAP_RANGES"] = _validate_gap_ranges(cfg["GAP_RANGES"])

    rsi_periods = _ensure_iterable_ints(cfg["RSI_PERIODS"], "RSI_PERIODS")
    if not rsi_periods:
        raise ValueError("RSI_PERIODS: list must not be empty")
    allowed = set(range(2, 15))
    if not set(rsi_periods).issubset(allowed):
        raise ValueError("RSI_PERIODS: all values must be within 2..14")
    cfg["RSI_PERIODS"] = sorted(set(rsi_periods))

    buy_th = _ensure_iterable_numbers(cfg["BUY_THRESHOLDS"], "BUY_THRESHOLDS")
    sell_th = _ensure_iterable_numbers(cfg["SELL_THRESHOLDS"], "SELL_THRESHOLDS")
    if not buy_th:
        raise ValueError("BUY_THRESHOLDS: must not be empty")
    if not sell_th:
        raise ValueError("SELL_THRESHOLDS: must not be empty")
    if min(buy_th) >= max(sell_th):
        raise ValueError("thresholds: min(BUY_THRESHOLDS) must be < max(SELL_THRESHOLDS)")
    cfg["BUY_THRESHOLDS"] = buy_th
    cfg["SELL_THRESHOLDS"] = sell_th

    if not isinstance(cfg["CSV_CACHE_FILE"], str):
        raise ValueError("CSV_CACHE_FILE: expected str (path to CSV)")

    if not isinstance(cfg["INITIAL_CASH"], (int, float)) or cfg["INITIAL_CASH"] <= 0:
        raise ValueError("INITIAL_CASH: expected number > 0")

    cfg.setdefault("BUFFER_DAYS", 30)
    cfg.setdefault("DAILY_CASH", 0.0)
    cfg.setdefault("RSI_TOLERANCE_PCT", 0.00001)
    cfg.setdefault("_NOTE", "Date ordering not validated in Phase 1")
    return cfg
