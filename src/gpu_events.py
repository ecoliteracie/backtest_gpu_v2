# src/gpu_events.py
from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np

try:
    from numba import cuda
    _NUMBA_IMPORT_OK = True
except Exception:
    cuda = None
    _NUMBA_IMPORT_OK = False

NONE, BUY, SELL = 0, 1, 2


def _prepare_rsi_by_period(df, close_map: Dict[int, str]):
    periods = sorted(close_map.keys())
    T = len(df)
    P = len(periods)
    arr = np.empty((P, T), dtype=np.float64)
    for i, p in enumerate(periods):
        arr[i, :] = df[close_map[p]].to_numpy(np.float64, copy=False)
    p2row = {p: i for i, p in enumerate(periods)}
    return arr, periods, p2row


def _numba_supported() -> tuple[bool, str]:
    if not _NUMBA_IMPORT_OK:
        return False, "Numba CUDA not importable"
    try:
        if not cuda.is_available():
            return False, "Numba CUDA reports unavailable"
        dev = cuda.get_current_device()
        cc = getattr(dev, "compute_capability", (0, 0))
        major, minor = int(cc[0]), int(cc[1])
        # Conservative: many Numba builds (as of 2025) don’t officially support cc 12.x yet.
        if major >= 12:
            return False, f"Unsupported compute capability {major}.{minor} for current Numba build"
        return True, f"cc={major}.{minor}"
    except Exception as e:
        return False, f"probe failed: {e}"


def _launch_numba_kernel(close, close_prev, rsi_by_period, regime_mask,
                         buy_idx, sell_idx, buy_thr, sell_thr) -> np.ndarray:
    C = buy_idx.shape[0]
    T = close.shape[0]

    events_type = np.zeros((C, T), dtype=np.int8)  # host buffer

    d_close      = cuda.to_device(close)
    d_close_prev = cuda.to_device(close_prev)
    d_rsiPT      = cuda.to_device(rsi_by_period)                  # (P,T)
    d_regime     = cuda.to_device(regime_mask.astype(np.bool_))   # (T,)
    d_buy_idx    = cuda.to_device(buy_idx.astype(np.int32))
    d_sell_idx   = cuda.to_device(sell_idx.astype(np.int32))
    d_buy_thr    = cuda.to_device(buy_thr.astype(np.float64))
    d_sell_thr   = cuda.to_device(sell_thr.astype(np.float64))
    d_events     = cuda.to_device(events_type)                    # (C,T)

    threads = 128
    blocks = (C + threads - 1) // threads

    @cuda.jit
    def _kernel(close_, close_prev_, rsiPT_, regime_, buy_idx_, sell_idx_, buy_thr_, sell_thr_, out_events_):
        i = cuda.grid(1)  # combo id
        C_ = out_events_.shape[0]
        T_ = close_.shape[0]
        if i >= C_:
            return

        p_buy  = buy_idx_[i]
        p_sell = sell_idx_[i]
        thr_b  = buy_thr_[i]
        thr_s  = sell_thr_[i]

        in_pos = False

        for d in range(1, T_):
            if not regime_[d]:
                continue

            cb  = rsiPT_[p_buy, d]
            cbp = rsiPT_[p_buy, d - 1]
            cs  = rsiPT_[p_sell, d]
            csp = rsiPT_[p_sell, d - 1]
            cl  = close_[d]
            clp = close_prev_[d]

            # NaN-safe (NaN != NaN)
            valid_buy  = (cb == cb) and (cbp == cbp) and (cl == cl) and (clp == clp)
            valid_sell = (cs == cs) and (csp == csp)

            if (not in_pos) and valid_buy:
                if (cbp < thr_b) and (cb < thr_b) and (cl > clp):
                    out_events_[i, d] = 1
                    in_pos = True
                    continue
            if in_pos and valid_sell:
                if (csp > thr_s) and (cs > thr_s):
                    out_events_[i, d] = 2
                    in_pos = False

    _kernel[blocks, threads](d_close, d_close_prev, d_rsiPT, d_regime,
                             d_buy_idx, d_sell_idx, d_buy_thr, d_sell_thr, d_events)

    # Catch device-side faults early
    cuda.synchronize()

    return d_events.copy_to_host()


def _cpu_fallback(close, close_prev, rsi_by_period, regime_mask, buy_idx, sell_idx, buy_thr, sell_thr) -> np.ndarray:
    C = buy_idx.shape[0]
    T = close.shape[0]
    events_type = np.zeros((C, T), dtype=np.int8)
    reg = regime_mask

    for i in range(C):
        p_b = int(buy_idx[i])
        p_s = int(sell_idx[i])
        thr_b = float(buy_thr[i])
        thr_s = float(sell_thr[i])
        r_b = rsi_by_period[p_b]
        r_s = rsi_by_period[p_s]

        in_pos = False
        for d in range(1, T):
            if not reg[d]:
                continue
            cbp, cb = r_b[d - 1], r_b[d]
            csp, cs = r_s[d - 1], r_s[d]
            c, cp = close[d], close_prev[d]

            valid_buy  = np.isfinite(cbp) and np.isfinite(cb) and np.isfinite(c) and np.isfinite(cp)
            valid_sell = np.isfinite(csp) and np.isfinite(cs)

            if (not in_pos) and valid_buy and (cbp < thr_b) and (cb < thr_b) and (c > cp):
                events_type[i, d] = BUY
                in_pos = True
                continue
            if in_pos and valid_sell and (csp > thr_s) and (cs > thr_s):
                events_type[i, d] = SELL
                in_pos = False

    return events_type


def build_event_streams(
    df,
    rsi_maps: Dict[str, Dict[int, str]],
    regime_mask_host: np.ndarray,
    combos: List[Tuple[int, int, float, float]],
    backend: str = "auto",
    batch_size: int | None = None,
) -> Dict[str, object]:
    """
    Build events_type[C,T] (0/1/2) for a list of combos.
    """
    T = len(df)
    close = df["Close"].to_numpy(np.float64, copy=False)
    close_prev = np.empty_like(close)
    close_prev[0] = np.nan
    close_prev[1:] = close[:-1]

    rsiPT, periods, p2row = _prepare_rsi_by_period(df, rsi_maps["close_map"])

    C = len(combos)
    buy_idx = np.empty(C, dtype=np.int32)
    sell_idx = np.empty(C, dtype=np.int32)
    buy_thr = np.empty(C, dtype=np.float64)
    sell_thr = np.empty(C, dtype=np.float64)
    for i, (bp, sp, bthr, sthr) in enumerate(combos):
        if bp not in p2row or sp not in p2row:
            raise ValueError(f"RSI period missing in maps: buy={bp} or sell={sp}")
        buy_idx[i] = p2row[bp]
        sell_idx[i] = p2row[sp]
        buy_thr[i] = bthr
        sell_thr[i] = sthr

    if regime_mask_host is None:
        regime_mask_host = np.ones(T, dtype=bool)
    if len(regime_mask_host) != T:
        raise ValueError("regime_mask_host length mismatch")

    # Decide backend
    use_numba = False
    reason = ""
    if backend.lower() == "numba":
        ok, reason = _numba_supported()
        use_numba = ok
    elif backend.lower() == "auto":
        ok, reason = _numba_supported()
        use_numba = ok
    else:
        use_numba = False
        reason = "forced CPU backend"

    if use_numba:
        try:
            events_type = _launch_numba_kernel(close, close_prev, rsiPT, regime_mask_host,
                                               buy_idx, sell_idx, buy_thr, sell_thr)
            return {"events_type": events_type, "meta": {"combos": combos, "T": T, "C": C, "backend": f"numba({reason})"}}
        except Exception as e:
            # Hard fallback to CPU
            events_type = _cpu_fallback(close, close_prev, rsiPT, regime_mask_host,
                                        buy_idx, sell_idx, buy_thr, sell_thr)
            return {"events_type": events_type, "meta": {"combos": combos, "T": T, "C": C, "backend": f"cpu(fallback:{e})"}}

    # CPU path
    events_type = _cpu_fallback(close, close_prev, rsiPT, regime_mask_host,
                                buy_idx, sell_idx, buy_thr, sell_thr)
    return {"events_type": events_type, "meta": {"combos": combos, "T": T, "C": C, "backend": f"cpu({reason})"}}
