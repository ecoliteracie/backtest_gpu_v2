# src/gpu_events.py
from __future__ import annotations

# --- TOP OF FILE (imports): add njit import ---
from typing import List, Tuple, Dict
import numpy as np

try:
    from numba import cuda
    _NUMBA_IMPORT_OK = True
except Exception:
    cuda = None
    _NUMBA_IMPORT_OK = False

# NEW: bring in numba.njit for fast CPU fallback
try:
    from numba import njit
    _NUMBA_NJIT_OK = True
except Exception:
    njit = None
    _NUMBA_NJIT_OK = False

NONE, BUY, SELL = 0, 1, 2



# Keep NONE, BUY, SELL = 0,1,2 as-is

def _prepare_rsi_by_period(df, map_dict):
    """Build a (P,T) array for the given period->column map, plus period order and a period->row index map."""
    periods = sorted(map_dict.keys())
    T = len(df)
    P = len(periods)
    arr = np.empty((P, T), dtype=np.float64)
    for i, p in enumerate(periods):
        col = map_dict[p]
        if col not in df.columns:
            raise ValueError(f"Required RSI column missing: {col}")
        arr[i, :] = df[col].to_numpy(np.float64, copy=False)
    p2row = {p: i for i, p in enumerate(periods)}
    return arr, periods, p2row

def _prepare_by_period(df, map_dict: Dict[int, str]) -> tuple[np.ndarray, list[int]]:
    periods = sorted(map_dict.keys())
    T = len(df)
    arr = np.empty((len(periods), T), dtype=np.float64)
    for i, p in enumerate(periods):
        col = map_dict[p]
        if col not in df.columns:
            raise ValueError(f"Required RSI column missing: {col}")
        arr[i, :] = df[col].to_numpy(np.float64, copy=False)
    return arr, periods

def _prepare_by_period_aligned(df, periods: list[int], map_dict: Dict[int, str]) -> np.ndarray:
    T = len(df)
    arr = np.empty((len(periods), T), dtype=np.float64)
    for i, p in enumerate(periods):
        col = map_dict.get(p)
        if col is None or col not in df.columns:
            raise ValueError(f"Required RSI column missing for period {p}")
        arr[i, :] = df[col].to_numpy(np.float64, copy=False)
    return arr

def _band_ok_buy_vals(low_val: float, high_val: float, thr: float) -> bool:
    # bracket OR entirely below threshold
    return ((low_val <= thr) and (thr <= high_val)) or (high_val < thr)

def _band_ok_sell_vals(low_val: float, high_val: float, thr: float) -> bool:
    # bracket OR entirely above threshold
    return ((low_val <= thr) and (thr <= high_val)) or (low_val > thr)


def _numba_supported() -> tuple[bool, str]:
    if not _NUMBA_IMPORT_OK:
        return False, "Numba CUDA not importable"
    try:
        if not cuda.is_available():
            return False, "Numba CUDA reports unavailable"
        dev = cuda.get_current_device()
        cc = getattr(dev, "compute_capability", (0, 0))
        major, minor = int(cc[0]), int(cc[1])
        # Conservative: many Numba builds (as of now) lack cc 12.x support
        if major >= 12:
            return False, f"compute capability {major}.{minor} unsupported by current Numba build"
        return True, f"cc={major}.{minor}"
    except Exception as e:
        return False, f"probe failed: {e}"



def _launch_numba_kernel(rsi_closePT, rsi_lowPT, rsi_highPT, regime_mask,
                         buy_idx, sell_idx, buy_thr, sell_thr) -> np.ndarray:
    C = buy_idx.shape[0]
    T = rsi_closePT.shape[1]
    events_type = np.zeros((C, T), dtype=np.int8)  # host buffer

    d_rsiC = cuda.to_device(rsi_closePT)                 # (P,T)
    d_rsiL = cuda.to_device(rsi_lowPT)                   # (P,T)
    d_rsiH = cuda.to_device(rsi_highPT)                  # (P,T)
    d_reg  = cuda.to_device(regime_mask.astype(np.bool_))
    d_bi   = cuda.to_device(buy_idx.astype(np.int32))
    d_si   = cuda.to_device(sell_idx.astype(np.int32))
    d_bt   = cuda.to_device(buy_thr.astype(np.float64))
    d_st   = cuda.to_device(sell_thr.astype(np.float64))
    d_evt  = cuda.to_device(events_type)

    threads = 128
    blocks = (C + threads - 1) // threads

    @cuda.jit
    def _kernel(rsiC, rsiL, rsiH, reg, bi, si, bt, st, out_evt):
        i = cuda.grid(1)
        C_ = out_evt.shape[0]
        T_ = rsiC.shape[1]
        if i >= C_:
            return

        p_b  = bi[i]
        p_s  = si[i]
        thrb = bt[i]
        thrs = st[i]

        in_pos = False

        for d in range(1, T_):
            if not reg[d]:
                continue

            # BUY side
            cbp = rsiC[p_b, d - 1]
            lb  = rsiL[p_b, d]
            hb  = rsiH[p_b, d]
            valid_buy = (cbp == cbp) and (lb == lb) and (hb == hb)  # NaN-safe

            if (not in_pos) and valid_buy:
                # prev close < thr AND (bracket OR all-below)
                if (cbp < thrb) and (((lb <= thrb) and (thrb <= hb)) or (hb < thrb)):
                    out_evt[i, d] = 1
                    in_pos = True
                    continue

            # SELL side
            csp = rsiC[p_s, d - 1]
            ls  = rsiL[p_s, d]
            hs  = rsiH[p_s, d]
            valid_sell = (csp == csp) and (ls == ls) and (hs == hs)

            if in_pos and valid_sell:
                # prev close > thr AND (bracket OR all-above)
                if (csp > thrs) and (((ls <= thrs) and (thrs <= hs)) or (ls > thrs)):
                    out_evt[i, d] = 2
                    in_pos = False

    _kernel[blocks, threads](d_rsiC, d_rsiL, d_rsiH, d_reg, d_bi, d_si, d_bt, d_st, d_evt)
    cuda.synchronize()
    return d_evt.copy_to_host()



def _cpu_fallback(rsi_close_by_period, rsi_low_by_period, rsi_high_by_period,
                  regime_mask, buy_idx, sell_idx, buy_thr, sell_thr) -> np.ndarray:
    C = buy_idx.shape[0]
    T = rsi_close_by_period.shape[1]
    events_type = np.zeros((C, T), dtype=np.int8)
    reg = regime_mask

    for i in range(C):
        p_b = int(buy_idx[i])
        p_s = int(sell_idx[i])
        thr_b = float(buy_thr[i])
        thr_s = float(sell_thr[i])

        r_close_b = rsi_close_by_period[p_b]
        r_low_b   = rsi_low_by_period[p_b]
        r_high_b  = rsi_high_by_period[p_b]

        r_close_s = rsi_close_by_period[p_s]
        r_low_s   = rsi_low_by_period[p_s]
        r_high_s  = rsi_high_by_period[p_s]

        in_pos = False
        for d in range(1, T):
            if not reg[d]:
                continue

            # BUY: prev close < thr & today's band brackets OR below thr
            cb_prev = r_close_b[d - 1]
            lb, hb  = r_low_b[d], r_high_b[d]
            if np.isfinite(cb_prev) and np.isfinite(lb) and np.isfinite(hb):
                if (not in_pos) and (cb_prev < thr_b) and _band_ok_buy_vals(lb, hb, thr_b):
                    events_type[i, d] = BUY
                    in_pos = True
                    continue

            # SELL: prev close > thr & today's band brackets OR above thr
            cs_prev = r_close_s[d - 1]
            ls, hs  = r_low_s[d], r_high_s[d]
            if in_pos and np.isfinite(cs_prev) and np.isfinite(ls) and np.isfinite(hs):
                if (cs_prev > thr_s) and _band_ok_sell_vals(ls, hs, thr_s):
                    events_type[i, d] = SELL
                    in_pos = False

    return events_type


# --- FAST CPU fallback using numba.njit (same logic as your validated CPU scan) ---
if _NUMBA_NJIT_OK:
    @njit(cache=True)
    def _cpu_fallback_njit(rsi_close_by_period, rsi_low_by_period, rsi_high_by_period,
                           regime_mask, buy_idx, sell_idx, buy_thr, sell_thr):
        C = buy_idx.shape[0]
        T = rsi_close_by_period.shape[1]
        events_type = np.zeros((C, T), np.int8)
        for i in range(C):
            p_b  = int(buy_idx[i])
            p_s  = int(sell_idx[i])
            thrb = float(buy_thr[i])
            thrs = float(sell_thr[i])
            in_pos = False
            for d in range(1, T):
                if not regime_mask[d]:
                    continue

                # BUY: prev close < thr & today's band brackets OR below thr
                cb_prev = rsi_close_by_period[p_b, d - 1]
                lb = rsi_low_by_period[p_b, d]
                hb = rsi_high_by_period[p_b, d]
                if (not in_pos) and (cb_prev == cb_prev) and (lb == lb) and (hb == hb):
                    if (cb_prev < thrb) and (((lb <= thrb) and (thrb <= hb)) or (hb < thrb)):
                        events_type[i, d] = BUY
                        in_pos = True
                        continue

                # SELL: prev close > thr & today's band brackets OR above thr
                cs_prev = rsi_close_by_period[p_s, d - 1]
                ls = rsi_low_by_period[p_s, d]
                hs = rsi_high_by_period[p_s, d]
                if in_pos and (cs_prev == cs_prev) and (ls == ls) and (hs == hs):
                    if (cs_prev > thrs) and (((ls <= thrs) and (thrs <= hs)) or (ls > thrs)):
                        events_type[i, d] = SELL
                        in_pos = False
        return events_type
else:
    # pure-Python fallback (slower) — unchanged from your validated logic
    def _cpu_fallback_njit(rsi_close_by_period, rsi_low_by_period, rsi_high_by_period,
                           regime_mask, buy_idx, sell_idx, buy_thr, sell_thr):
        C = buy_idx.shape[0]
        T = rsi_close_by_period.shape[1]
        events_type = np.zeros((C, T), np.int8)
        for i in range(C):
            p_b  = int(buy_idx[i])
            p_s  = int(sell_idx[i])
            thrb = float(buy_thr[i])
            thrs = float(sell_thr[i])
            in_pos = False
            for d in range(1, T):
                if not regime_mask[d]:
                    continue
                cb_prev = rsi_close_by_period[p_b, d - 1]
                lb = rsi_low_by_period[p_b, d]
                hb = rsi_high_by_period[p_b, d]
                if (not in_pos) and np.isfinite(cb_prev) and np.isfinite(lb) and np.isfinite(hb):
                    if (cb_prev < thrb) and (((lb <= thrb) and (thrb <= hb)) or (hb < thrb)):
                        events_type[i, d] = BUY
                        in_pos = True
                        continue
                cs_prev = rsi_close_by_period[p_s, d - 1]
                ls = rsi_low_by_period[p_s, d]
                hs = rsi_high_by_period[p_s, d]
                if in_pos and np.isfinite(cs_prev) and np.isfinite(ls) and np.isfinite(hs):
                    if (cs_prev > thrs) and (((ls <= thrs) and (thrs <= hs)) or (ls > thrs)):
                        events_type[i, d] = SELL
                        in_pos = False
        return events_type




from concurrent.futures import ProcessPoolExecutor, as_completed
import math

def _cpu_fallback_chunk(args) -> np.ndarray:
    # Expected 8-tuple: (rsi_close_PT, rsi_low_PT, rsi_high_PT, regime_mask,
    #                    buy_idx_chunk, sell_idx_chunk, buy_thr_chunk, sell_thr_chunk)
    (rsi_close_PT, rsi_low_PT, rsi_high_PT,
     regime_mask, buy_idx_chunk, sell_idx_chunk, buy_thr_chunk, sell_thr_chunk) = args

    return _cpu_fallback(
        rsi_close_PT, rsi_low_PT, rsi_high_PT,
        regime_mask, buy_idx_chunk, sell_idx_chunk, buy_thr_chunk, sell_thr_chunk
    )

# --- build_event_streams: explicit backend, robust fallbacks, no signature mismatches ---
def build_event_streams(
    df,
    rsi_maps: Dict[str, Dict[int, str]],
    regime_mask_host: np.ndarray,
    combos: List[Tuple[int, int, float, float]],
    backend: str = "auto",
    batch_size: int | None = None,
) -> Dict[str, object]:
    T = len(df)
    if regime_mask_host is None:
        regime_mask_host = np.ones(T, dtype=bool)
    if len(regime_mask_host) != T:
        raise ValueError("regime_mask_host length mismatch")

    # (P,T) arrays; CLOSE defines period indexing
    rsi_close_PT, close_periods, p2row = _prepare_rsi_by_period(df, rsi_maps["close_map"])
    rsi_low_PT,  _, _ = _prepare_rsi_by_period(df, rsi_maps["low_map"])
    rsi_high_PT, _, _ = _prepare_rsi_by_period(df, rsi_maps["high_map"])

    # combo → indices/threshold arrays
    C = len(combos)
    buy_idx = np.empty(C, dtype=np.int32)
    sell_idx = np.empty(C, dtype=np.int32)
    buy_thr  = np.empty(C, dtype=np.float64)
    sell_thr = np.empty(C, dtype=np.float64)
    for i, (bp, sp, bthr, sthr) in enumerate(combos):
        if bp not in p2row or sp not in p2row:
            raise ValueError(f"RSI period missing in maps: buy={bp} or sell={sp}")
        buy_idx[i] = p2row[bp]
        sell_idx[i] = p2row[sp]
        buy_thr[i]  = bthr
        sell_thr[i] = sthr

    # Decide backend
    bmeta = ""
    use_numba = False
    if backend.lower() == "numba":
        ok, reason = _numba_supported()
        use_numba = ok
        bmeta = f"numba({reason})" if ok else f"cpu_njit({reason})"
    elif backend.lower() == "auto":
        ok, reason = _numba_supported()
        use_numba = ok
        bmeta = f"numba({reason})" if ok else f"cpu_njit({reason})"
    else:
        bmeta = "cpu_njit(forced)"

    # Try GPU kernel if allowed; else fast CPU
    if use_numba:
        try:
            events_type = _launch_numba_kernel(
                rsi_close_PT, rsi_low_PT, rsi_high_PT,
                regime_mask_host, buy_idx, sell_idx, buy_thr, sell_thr
            )
            return {"events_type": events_type,
                    "meta": {"combos": combos, "T": T, "C": C, "backend": bmeta}}
        except Exception as e:
            # Explicit fallback message in backend field
            bmeta = f"cpu_njit(fallback:{e})"

    # Fast CPU (njit) path
    events_type = _cpu_fallback_njit(
        rsi_close_PT, rsi_low_PT, rsi_high_PT,
        regime_mask_host, buy_idx, sell_idx, buy_thr, sell_thr
    )
    return {"events_type": events_type,
            "meta": {"combos": combos, "T": T, "C": C, "backend": bmeta}}
