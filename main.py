# main.py — Phases 1..6K with externalized banners
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import src.regimes as regimes

from src.logging_setup import get_logger
from src.config_loader import load, validate
from src.io_csv import resolve_csv_path, probe_csv, load_prices
from src.validate import require_columns
from src.windowing import compute_requested_window, trim_for_backtest
from src.benchmarks import buy_and_hold
from src.gpu_backend import select_backend, sanity_compute_check, select_backend    
from src.banners import phase1, phase2, phase3, phase4, phase5, phase6k, phase7, phase8, phase9, phase10
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

        # IMPORTANT: do not shadow regimes.regime_mask()
        REGIME_MASK_HOST = None  # optional; Phase 9 will provide one later

        backend = select_backend("cupy")  # dict with 'xp'
        cp = backend["xp"]

        close_map = RSI_MAPS["close_map"]  # from Phase 7
        buy_ok_dev, sell_ok_dev, meta = make_buy_sell_masks(
            df=df,
            close_map=close_map,
            buy_period=buy_period,
            sell_period=sell_period,
            buy_thr=buy_thr,
            sell_thr=sell_thr,
            regime_mask_host=REGIME_MASK_HOST,
            cp=backend,  # pass backend dict or cp; both supported
        )

        banner8 = phase8.build_banner(meta)
        print()
        print(banner8)
        for line in banner8.splitlines():
            logger8.info(line)

        GPU_BUY_MASK = buy_ok_dev
        GPU_SELL_MASK = sell_ok_dev

    except Exception as e:
        msg = f"Phase 8 failed: {e}"
        print(msg)
        logger8.error(msg)
        sys.exit(1)
    # ---- End Phase 8 ----


    log9_path = Path("logs") / "phase09_regimes.log"
    logger9 = get_logger("phase09", log9_path)
    
    try:
        gap_ranges = cfg.get("GAP_RANGES", [])
        if not isinstance(gap_ranges, (list, tuple)) or not gap_ranges:
            raise ValueError("GAP_RANGES must be a non-empty list of (low, high) tuples.")
        for rng in gap_ranges:
            if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
                raise ValueError(f"Invalid GAP_RANGES entry: {rng!r} (expected 2-tuple)")

        # Compute MA_GAP and attach
        ma_gap = regimes.compute_ma_gap(df)
        if ma_gap.dtype != "float64":
            raise TypeError(f"MA_GAP must be float64, got {ma_gap.dtype}")
        if not np.isfinite(ma_gap.dropna().to_numpy()).all():
            raise ValueError("MA_GAP contains inf or -inf values after denominator guard.")

        # Label by ranges and attach
        reg = regimes.label_by_ranges(df, gap_ranges)
        labels = regimes.generate_regime_labels(gap_ranges)

        # Counts and samples
        counts = {lab: int((reg == lab).sum()) for lab in labels}
        samples: dict[str, list[str]] = {}
        for lab in labels:
            idx = df.index[(reg == lab)]
            samples[lab] = [str(d) for d in idx[:3]]

        non_null = int(ma_gap.notna().sum())
        total = int(ma_gap.shape[0])
        fv_idx = ma_gap.first_valid_index()
        first_valid = str(fv_idx) if fv_idx is not None else None

        banner9 = phase9.build_banner(
            gap_ranges=gap_ranges,
            ma_gap_dtype=str(ma_gap.dtype),
            non_null=non_null,
            total=total,
            first_valid=first_valid,
            labels=labels,
            counts=counts,
            samples=samples,
        )
        print()
        print(banner9)
        for line in banner9.splitlines():
            logger9.info(line)

        # Choose a regime and build a host mask for later phases
        mid_idx = len(labels) // 2
        CHOSEN_REGIME_LABEL = labels[mid_idx] if labels else "gap_all"

        # Guard against accidental shadowing: regimes.regime_mask must be callable
        if not callable(getattr(regimes, "regime_mask", None)):
            raise RuntimeError("Phase 9 internal error: regimes.regime_mask is not callable (name shadowed).")

        REGIME_MASK_HOST = regimes.regime_mask(df, CHOSEN_REGIME_LABEL)
        REGIME_LABELS = labels

    except Exception as e:
        msg = f"Phase 9 failed: {e}"
        print(msg)
        logger9.error(msg)
        sys.exit(1)
    # ---- End Phase 9 ----


    log10_path = Path("logs") / "phase10_masks_by_regime.log"
    logger10 = get_logger("phase10", log10_path)

    try:
        # Parameters (single combo, as in Phase 8)
        rsi_periods = list(sorted(cfg.get("RSI_PERIODS", [2])))
        buy_period  = int(rsi_periods[0])
        sell_period = int(rsi_periods[0])
        buy_thr  = float(min(cfg.get("BUY_THRESHOLDS", [24])))
        sell_thr = float(max(cfg.get("SELL_THRESHOLDS", [90])))

        # GPU backend
        try:
            cp  # noqa: F821
        except NameError:
            backend = select_backend("cupy")
            cp = backend["xp"]
        else:
            try:
                backend = select_backend("cupy")
            except Exception:
                backend = {"xp": cp, "device_name": "Unknown", "cc": "?"}

        # Ensure masks exist (reuse Phase 8 or build now)
        try:
            GPU_BUY_MASK  # noqa: F821
            GPU_SELL_MASK # noqa: F821
        except NameError:
            close_map = RSI_MAPS["close_map"]
            GPU_BUY_MASK, GPU_SELL_MASK, meta8 = make_buy_sell_masks(
                df=df,
                close_map=close_map,
                buy_period=buy_period,
                sell_period=sell_period,
                buy_thr=buy_thr,
                sell_thr=sell_thr,
                regime_mask_host=None,
                cp=backend,
            )

        # 1) Totals on all rows
        buy_all  = int(cp.count_nonzero(GPU_BUY_MASK).item())
        sell_all = int(cp.count_nonzero(GPU_SELL_MASK).item())

        # 2) Restrict to MA_GAP domain (non-NaN)
        if "MA_GAP" not in df.columns:
            raise RuntimeError("Phase 10 requires df['MA_GAP'] from Phase 9.")

        in_domain_host = (~df["MA_GAP"].isna()).to_numpy(dtype=bool, copy=False)
        assert in_domain_host.shape[0] == len(df.index), "in_domain mask misaligned"
        in_domain_dev  = cp.asarray(in_domain_host, dtype=cp.bool_)

        buy_dom  = int(cp.count_nonzero(GPU_BUY_MASK & in_domain_dev).item())
        sell_dom = int(cp.count_nonzero(GPU_SELL_MASK & in_domain_dev).item())
        buy_nan  = buy_all - buy_dom
        sell_nan = sell_all - sell_dom

        # 3) Per-regime counts & samples (only within domain)
        labels = REGIME_LABELS if 'REGIME_LABELS' in globals() else regimes.generate_regime_labels(cfg.get("GAP_RANGES", []))
        per_blocks = []
        sum_buy = 0
        sum_sell = 0

        # Build and retain host masks to run integrity checks
        host_masks: dict[str, np.ndarray] = {}
        for lab in labels:
            m_host = regimes.regime_mask(df, lab)  # excludes NaN domain by construction
            if m_host.dtype != bool:
                m_host = m_host.astype(bool, copy=False)
            assert m_host.shape[0] == len(df.index), f"regime mask misaligned for {lab}"
            host_masks[lab] = m_host

            dev_mask = cp.asarray(m_host, dtype=cp.bool_)
            # constrain explicitly to domain for samples (redundant but clear)
            dev_mask &= in_domain_dev

            buy_reg  = GPU_BUY_MASK & dev_mask
            sell_reg = GPU_SELL_MASK & dev_mask

            cnt_buy  = int(cp.count_nonzero(buy_reg).item())
            cnt_sell = int(cp.count_nonzero(sell_reg).item())
            sum_buy  += cnt_buy
            sum_sell += cnt_sell

            def _dates_from_mask(mask_dev, k=5):
                idx_dev = cp.where(mask_dev)[0]
                if idx_dev.size == 0:
                    return []
                idx_host = cp.asnumpy(idx_dev[:k]).tolist()
                return [str(df.index[i]) for i in idx_host]

            samples_buy  = _dates_from_mask(buy_reg, 5)
            samples_sell = _dates_from_mask(sell_reg, 5)

            per_blocks.append({
                "label": lab,
                "counts": {"buy_ok": cnt_buy, "sell_ok": cnt_sell},
                "samples": {"buy": samples_buy, "sell": samples_sell},
            })

        # 4) Optional "NaN regime" diagnostics (signals outside MA_GAP domain)
        per_nan = None
        if buy_nan > 0 or sell_nan > 0:
            nan_mask_dev = ~in_domain_dev
            buy_nan_mask = GPU_BUY_MASK & nan_mask_dev
            sell_nan_mask = GPU_SELL_MASK & nan_mask_dev

            def _dates_from_mask(mask_dev, k=5):
                idx_dev = cp.where(mask_dev)[0]
                if idx_dev.size == 0:
                    return []
                idx_host = cp.asnumpy(idx_dev[:k]).tolist()
                return [str(df.index[i]) for i in idx_host]

            samples_buy_nan  = _dates_from_mask(buy_nan_mask, 5)
            samples_sell_nan = _dates_from_mask(sell_nan_mask, 5)
            per_nan = {
                "label": "gap_(NaN)",
                "counts": {"buy_ok": buy_nan, "sell_ok": sell_nan},
                "samples": {"buy": samples_buy_nan, "sell": samples_sell_nan},
            }
            per_blocks.append(per_nan)

        # 5) Integrity checks (host-side)
        # Overlaps: rows assigned to more than one regime (should be zero)
        n = len(df.index)
        overlaps_mask_total = np.zeros(n, dtype=bool)
        first_overlap_date = None
        labs = list(labels)
        for i in range(len(labs)):
            mi = host_masks[labs[i]]
            for j in range(i + 1, len(labs)):
                mj = host_masks[labs[j]]
                ov = mi & mj
                if ov.any():
                    overlaps_mask_total |= ov
                    if first_overlap_date is None:
                        first_overlap_date = str(df.index[np.argmax(ov)])
        overlap_count = int(overlaps_mask_total.sum())

        # Holes: domain rows not covered by any regime
        union_mask = np.zeros(n, dtype=bool)
        for lab in labs:
            union_mask |= host_masks[lab]
        hole_mask = in_domain_host & (~union_mask)
        hole_count = int(hole_mask.sum())
        first_hole_date = str(df.index[np.argmax(hole_mask)]) if hole_count > 0 else None

        # Boundary audit: verify [low, high) assignment at edges
        gr = cfg.get("GAP_RANGES", [])
        boundaries = sorted({float(b) for ab in gr for b in ab if b is not None})
        boundary_assignments: dict[str, dict[str, int]] = {}
        if boundaries:
            ma_gap = df["MA_GAP"].to_numpy(dtype=np.float64, copy=False)
            reg_series = df["REGIME"].astype("object")
            labels_canon = regimes.generate_regime_labels(gr)
            for b in boundaries:
                # rows "equal" to boundary (tolerance)
                at_b = np.isclose(ma_gap, b, atol=1e-12, rtol=0.0)
                dest_counts: dict[str, int] = {}
                if at_b.any():
                    # Count by actual label
                    for lab in labels_canon:
                        dest_counts[lab] = int(((reg_series == lab).to_numpy() & at_b).sum())
                boundary_assignments[str(b)] = dest_counts

        # 6) Compose banner
        meta10 = {
            "params": {"buy_period": buy_period, "sell_period": sell_period, "buy_thr": buy_thr, "sell_thr": sell_thr},
            "totals": {"buy_all": buy_all, "sell_all": sell_all, "buy_dom": sum_buy*0 + buy_dom, "sell_dom": sum_sell*0 + sell_dom,
                    "buy_nan": buy_nan, "sell_nan": sell_nan},
            "backend": {"name": backend.get("device_name", "Unknown"), "cc": backend.get("cc", "?")},
            "per_regime": per_blocks,
            "checks": {
                "sum_buy": sum_buy, "sum_sell": sum_sell,
                "overlaps": {"count": overlap_count, "first_date": first_overlap_date},
                "holes":    {"count": hole_count,   "first_date": first_hole_date},
                "boundary_assignments": boundary_assignments,
            },
        }

        banner10 = phase10.build_banner(meta10)
        print()
        print(banner10)
        for line in banner10.splitlines():
            logger10.info(line)

    except Exception as e:
        msg = f"Phase 10 failed: {e}"
        print(msg)
        logger10.error(msg)
        sys.exit(1)
    # ---- End Phase 10 ----




    return 0


if __name__ == "__main__":
    sys.exit(main())
