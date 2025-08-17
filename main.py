# main.py — Phases 1..6K with externalized banners
from __future__ import annotations

import sys
from pathlib import Path

from src.logging_setup import get_logger
from src.config_loader import load, validate
from src.io_csv import resolve_csv_path, probe_csv, load_prices
from src.validate import require_columns
from src.windowing import compute_requested_window, trim_for_backtest
from src.benchmarks import buy_and_hold
from src.gpu_backend import select_backend, sanity_compute_check, select_backend
from src.banners import phase1, phase2, phase3, phase4, phase5, phase6k, phase7, phase8
from src.columns import detect_rsi_columns, analyze_rsi_invariants
from src.signals_gpu import make_buy_sell_masks


def _ensure_dirs() -> None:
    for d in ("logs", "results"):
        Path(d).mkdir(parents=True, exist_ok=True)


def main() -> int:
    _ensure_dirs()

    # Phase 1
    log1 = Path("logs") / "phase01_init.log"
    logger1 = get_logger("phase01", log1)
    try:
        cfg = load("configs.default")
        cfg = validate(cfg)
    except Exception as e:
        msg = f"CONFIG ERROR: {e}"
        print(msg)
        logger1.error(msg)
        return 1
    banner1 = phase1.build_banner(cfg)
    print(banner1)
    for line in banner1.splitlines():
        logger1.info(line)

    # Phase 2
    log2 = Path("logs") / "phase02_csv_probe.log"
    logger2 = get_logger("phase02", log2)
    try:
        csv_abs = resolve_csv_path(cfg, base_dir=str(Path(__file__).resolve().parent))
        probe = probe_csv(csv_abs, sample_rows=5)
    except Exception as e:
        msg = f"CSV probe failed: {e}"
        print(msg)
        logger2.error(msg)
        return 1
    banner2 = phase2.build_banner(probe)
    print()
    print(banner2)
    for line in banner2.splitlines():
        logger2.info(line)

    # Phase 3
    log3 = Path("logs") / "phase03_df_load.log"
    logger3 = get_logger("phase03", log3)
    try:
        df = load_prices(csv_abs)
        require_columns(df, ["Open", "High", "Low", "Close"])
    except Exception as e:
        msg = f"DataFrame load/validate failed: {e}"
        print(msg)
        logger3.error(msg)
        return 1
    banner3 = phase3.build_banner(df)
    print()
    print(banner3)
    for line in banner3.splitlines():
        logger3.info(line)

    # Phase 4
    log4 = Path("logs") / "phase04_window.log"
    logger4 = get_logger("phase04", log4)
    try:
        start_ts, end_ts, buffer_days = compute_requested_window(cfg)
        max_period = max(cfg["RSI_PERIODS"])
        df_trim, effective_start, buffer_ok = trim_for_backtest(
            df, start_ts, end_ts, buffer_days, max_period
        )
        if df_trim.shape[0] == 0:
            msg = "No rows in trimmed window; check your date range and buffer."
            print(msg)
            logger4.error(msg)
            return 1
    except Exception as e:
        msg = f"Windowing failed: {e}"
        print(msg)
        logger4.error(msg)
        return 1
    banner4 = phase4.build_banner(
        df=df,
        df_trim=df_trim,
        requested_start=start_ts,
        requested_end=end_ts,
        effective_start=effective_start,
        buffer_ok=buffer_ok,
        max_period=max_period,
    )
    print()
    print(banner4)
    for line in banner4.splitlines():
        logger4.info(line)

    # Phase 5
    log5 = Path("logs") / "phase05_bh.log"
    logger5 = get_logger("phase05", log5)
    try:
        bh = buy_and_hold(df_trim, float(cfg["INITIAL_CASH"]))
    except Exception as e:
        msg = f"Buy-and-hold failed: {e}"
        print(msg)
        logger5.error(msg)
        return 1
    banner5 = phase5.build_banner(bh, initial_cash=float(cfg["INITIAL_CASH"]))
    print()
    print(banner5)
    for line in banner5.splitlines():
        logger5.info(line)

    # Phase 6K — GPU Backend Probe
    log6 = Path("logs") / "phase06_gpu_probe.log"
    logger6 = get_logger("phase06k", log6)
    try:
        info = select_backend(preferred="cupy")
        sanity = sanity_compute_check(info["xp"])
        if not (sanity["abs_err"] < 1e-12):
            msg = f"GPU ERROR: Sanity compute mismatch: abs_err={sanity['abs_err']:.3e} exceeded 1e-12."
            print(msg)
            logger6.error(msg)
            return 1
    except SystemExit as e:
        msg = str(e)
        print(msg)
        logger6.error(msg)
        return 1
    except Exception as e:
        msg = f"GPU ERROR: Unexpected failure during GPU probe: {e}"
        print(msg)
        logger6.error(msg)
        return 1

    banner6 = phase6k.build_banner(info, sanity)
    print()
    print(banner6)
    for line in banner6.splitlines():
        logger6.info(line)

    log7_path = Path("logs") / "phase07_rsi_columns.log"
    logger7 = get_logger("phase07", log7_path)

    try:
        # df should be the trimmed DataFrame from prior phases
        rsi_maps = detect_rsi_columns(df)
        diag = analyze_rsi_invariants(df, rsi_maps)

        banner7 = phase7.build_banner(rsi_maps, diag)
        print()
        print(banner7)
        for line in banner7.splitlines():
            logger7.info(line)

        if not diag["bounds"]["ok"]:
            msg = f"Phase 7 failed: {diag['bounds']['reason']}"
            print(msg)
            logger7.error(msg)
            sys.exit(1)

        # Expose for downstream phases
        RSI_MAPS = rsi_maps  # keep in scope; pass to Phase 8 consumer(s)
    except Exception as e:
        msg = f"Phase 7 failed: {e}"
        print(msg)
        logger7.error(msg)
        sys.exit(1)
    # ---- End Phase 7 ----


    log8_path = Path("logs") / "phase08_masks_gpu.log"
    logger8 = get_logger("phase08", log8_path)

    try:
        # Resolve single test combo from config
        rsi_periods = list(sorted(cfg.get("RSI_PERIODS", [2])))
        buy_period  = int(rsi_periods[0])
        sell_period = int(rsi_periods[0])

        buy_thr  = float(min(cfg.get("BUY_THRESHOLDS", [24])))
        sell_thr = float(max(cfg.get("SELL_THRESHOLDS", [90])))

        regime_mask = None  # optional, will add in Phase 9

        # GPU backend: select_backend returns a dict with CuPy as 'xp'
        backend = select_backend("cupy")
        cp = backend["xp"]

        # Build masks (you may pass either 'cp' or the whole 'backend' dict; both now work)
        close_map = RSI_MAPS["close_map"]  # produced in Phase 7
        buy_ok_dev, sell_ok_dev, meta = make_buy_sell_masks(
            df=df,
            close_map=close_map,
            buy_period=buy_period,
            sell_period=sell_period,
            buy_thr=buy_thr,
            sell_thr=sell_thr,
            regime_mask_host=regime_mask,
            cp=backend,  # pass backend dict so banner can use device_name/cc directly
        )

        # Banner
        banner8 = phase8.build_banner(meta)
        print()
        print(banner8)
        for line in banner8.splitlines():
            logger8.info(line)

        # Keep device masks for later phases
        GPU_BUY_MASK = buy_ok_dev
        GPU_SELL_MASK = sell_ok_dev

    except Exception as e:
        msg = f"Phase 8 failed: {e}"
        print(msg)
        logger8.error(msg)
        sys.exit(1)
    # ---- End Phase 8 ----




    return 0


if __name__ == "__main__":
    sys.exit(main())
