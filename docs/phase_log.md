# RSI Backtest GPU Project — Phase Log

This document records the progressive development of the RSI backtesting engine, phase by phase.  
Each entry notes the **Goal**, **Changes**, and **Verification** steps.  
Future phases should be appended here in the same format.

---

## Phase 1 — Bootstrap & Config Load

**Goal**  
- Establish minimal workspace structure.  
- Import `configs/default.py` safely.  
- Validate required configuration keys.  
- Print/log a concise config summary.

**Changes**  
- Added `src/logging_setup.py` → `get_logger()` for stdout + file logging.  
- Added `src/config_loader.py` → `load()` and `validate()`.  
- Added `main.py` orchestration.  
- Created directories: `logs/`, `results/`.  

**Verification**  
- Ran `python main.py`.  
- Confirmed `[CONFIG] OK` banner on console.  
- Log written to `logs/phase01_init.log`.  
- Correct handling of nested `SETTINGS` dict in `default.py`.  

---

## Phase 2 — CSV Presence & Header Probe

**Goal**  
- Verify CSV exists and is readable.  
- Report size, line count, header, and first few rows without pandas.  
- Log the same summary.

**Changes**  
- Added `src/io_csv.py` →  
  - `resolve_csv_path()` resolves relative paths to `data/`.  
  - `probe_csv()` reports stats, header, sample lines.  
- Updated `main.py` → Phase 2 orchestration and log to `logs/phase02_csv_probe.log`.

**Verification**  
- Ran with `SOXL_full_ohlc_indicators.csv`.  
- Console showed absolute path, size (~4.6 MB), ~3775 lines.  
- Header printed with OHLC, RSI, MA, MACD, ATR columns.  
- 5 sample rows displayed.  
- Log file matched console output.  

---

## Phase 3 — DataFrame Load & Column Sanity

**Goal**  
- Load CSV into pandas DataFrame with strict dtypes.  
- Ensure Date index is datetime, unique, monotonic.  
- Verify required OHLC columns.  
- Summarize indicators present.

**Changes**  
- Updated `src/io_csv.py` → `load_prices()` (pandas loader).  
- Added `src/validate.py` →  
  - `require_columns()` checks OHLC presence.  
  - `summarize_indicators()` detects RSI_CLOSE periods, MA flags, preview extras.  
- Updated `main.py` → Phase 3 orchestration and log to `logs/phase03_df_load.log`.

**Verification**  
- Loaded ~3774 rows × ~96 cols into DataFrame.  
- Reported memory usage (~2.8 MB).  
- Date range 2010-07-19 → 2025-07-18.  
- Null counts for OHLC = 0.  
- RSI_CLOSE periods detected: 2–14.  
- MA_50 and MA_200 present.  
- Extras preview included MACD/ATR.  
- Printed 3 head and 3 tail rows.  
- Log matched console.  

---

## Phase 4 — Date Window Trim & Buffer Check

**Goal**  
- Trim DataFrame to `[START_DATE - BUFFER_DAYS, END_DATE]`.  
- Verify warm-up buffer sufficiency against `max(RSI_PERIODS)`.  
- Report effective ranges, row counts, and buffer status.

**Changes**  
- Added `src/windowing.py` →  
  - `parse_iso_date()` strict ISO parser.  
  - `compute_requested_window()` → parse start/end/buffer.  
  - `trim_for_backtest()` → slice DataFrame, assess buffer rows.  
- Updated `main.py` → Phase 4 orchestration and log to `logs/phase04_window.log`.

**Verification**  
- Request: start=2025-01-01, end=2025-07-01, buffer_days=30.  
- Effective range: 2024-12-02 → 2025-07-01.  
- Data range: 2010-07-19 → 2025-07-18.  
- Trimmed range: 2024-12-02 → 2025-07-01.  
- Row counts reported: total=3774, trimmed=145, dropped_pre=3629, dropped_post=0.  
- Buffer check: warmup rows=31, required=14, ok=True.  
- Log matched console output.  


## Phase 5 — Buy-and-Hold Baseline
**Goal**
* Establish a deterministic baseline for later GPU strategy comparisons.
* Simulate buying at the first trimmed close and holding through the last trimmed close.
* Report final value, ROI, and CAGR.
* Persist results to log for reproducibility.

**Changes**
* Added `src/benchmarks.py` → `buy_and_hold()` implementation.
* Updated `main.py` → Phase 5 orchestration.
* Created log file `logs/phase05_bh.log`.

**Verification**
* Ran on trimmed SOXL dataset (2024-12-02 → 2025-07-01).
* Start close ≈ 0.5833, end close ≈ 0.6277.
* Initial cash 1,000 → final ≈ 1,076.14.
* ROI ≈ +7.61%.
* CAGR ≈ +13.23% over 211 calendar days.
* Console banner and `logs/phase05_bh.log` matched exactly.

## Phase 6K — GPU Backend Probe

**Goal**

* Verify CUDA availability via CuPy on Windows 11.
* Capture device properties (name, compute capability, memory) and runtime/driver versions.
* Run a tiny on-GPU compute and cross-check against CPU to confirm numerical correctness.

**Changes**

* Added `src/gpu_backend.py` →

  * `select_backend("cupy")` probes device count, properties, versions, and runs a minimal device allocation/compute.
  * `sanity_compute_check(cp)` performs `sum(sqrt(linspace(0,1,1024)))` on GPU and CPU (float64) and compares.
  * `format_bytes()` helper for human-readable memory figures.
* Added `src/banners/phase6k.py` for the Phase 6K banner.
* Updated `main.py` → Phase 6K orchestration and logging to `logs/phase06_gpu_probe.log`.

**Verification**

* Ran `python main.py` after Phase 5.
* Console displayed “Phase 6K — GPU Backend Probe” banner.
* CuPy successfully imported; at least one CUDA device detected.
* Reported backend=cupy, device name and id, compute capability, total/free memory, driver/runtime versions, and CuPy version.
* Sanity compute passed: absolute error < 1e-12 between GPU and CPU results.
* Exit code 0; `logs/phase06_gpu_probe.log` matched the console banner.



## Phase 7 — RSI Column Binding & Invariants

**Goal**

* Bind precomputed RSI columns (periods 2..14; CLOSE required, LOW/HIGH/OPEN optional).
* Assert bounds and report diagnostics so downstream GPU mask logic can trust inputs.

**Changes**

* Added `src/columns.py` with:

  * `detect_rsi_columns(df)` to discover RSI columns and return `close_map`, `low_map`, `high_map`, `open_map`, `periods`, and `missing_variants`.
  * `analyze_rsi_invariants(df, rsi_maps)` for presence, bounds, NaN, ordering, and dtype diagnostics.
* Added `src/banners/phase7.py` to render a one-screen banner.
* Updated `main.py` to run Phase 7 after Phase 6K, mirror console → `logs/phase07_rsi_columns.log`, and expose `RSI_MAPS` to later phases.
* Hard-fail if any detected RSI column has values outside `[0,100]` beyond warm-up.

**Invariants Checked**

* Presence: at least one CLOSE-based RSI period in 2..14.
* Bounds (HARD): all detected RSI values within `[0,100]` after the warm-up NaNs.
* NaN accounting: total and head run-length per RSI column.
* Ordering (INFO): when `Low ≤ Open ≤ Close ≤ High` and all RSI variants exist, check `RSI_LOW ≤ RSI_OPEN ≤ RSI_CLOSE ≤ RSI_HIGH` on up to 200 samples/period.
* Dtypes: warn if any RSI column is not `float64` (no mutation in Phase 7).

**Result Snapshot (SOXL run)**

* Periods detected: `2..14 (count=13)`
* CLOSE map: `RSI_2_CLOSE, ..., RSI_14_CLOSE`
* LOW/HIGH/OPEN: present for all periods; missing: none
* NaNs: head run-lengths consistent with period (e.g., `RSI_2_CLOSE head=1`, `RSI_14_CLOSE head=13`)
* Bounds: All within `[0,100]` beyond warm-up
* Ordering: 0 violations across sampled rows for p=2..14
* Log: `logs/phase07_rsi_columns.log`

**Verification**

* Ran `python main.py` on SOXL dataset.
* Console output matched the log line-for-line.
* Manual spot checks confirmed the head NaN run-length equals the period minus one.

**Commit Message**

Phase 7: Bind RSI columns and assert invariants

- Add src/columns.py with detect_rsi_columns() and analyze_rsi_invariants()
- Add Phase-7 banner and file logging
- Bounds, NaN, and ordering diagnostics
- Expose RSI_MAPS for Phase 8 consumers



## Phase 8 — GPU Buy/Sell Predicate Masks

**Goal**

* Build buy/sell predicate masks on GPU (CuPy) from CLOSE-based RSI with two-day confirmation rules and buy-side momentum.
* Produce device masks and human-readable diagnostics in a per-phase log.

**Changes**

* Added `src/signals_gpu.py` with:

  * `make_buy_sell_masks(df, close_map, buy_period, sell_period, buy_thr, sell_thr, regime_mask_host=None, cp=None)`
  * Fully vectorized device predicates:

    * Buy: `(RSI_buy[d-1] < buy_thr) & (RSI_buy[d] < buy_thr) & (Close[d] > Close[d-1])`
    * Sell: `(RSI_sell[d-1] > sell_thr) & (RSI_sell[d] > sell_thr)`
  * NaN-safety: any NaN in `[d-1, d]` collapses predicate at `d` to False; day 0 forced False.
  * Regime mask (optional): host bool array intersected on device.
  * Returns two device boolean arrays and a `meta` dict with counts, first few dates, params, and backend info.
  * Backend compatibility: accepts either a CuPy module or the backend dict from `gpu_backend.select_backend("cupy")`.
* Added `src/banners/phase8.py` to render a one-screen banner with params, counts, sample dates, regime summary, and backend info.
* Updated `src/banners/__init__.py` to re-export `phase8`.
* Updated `main.py` to:

  * Read a single test combo from config (e.g., `buy_period=2`, `sell_period=2`, `buy_thr=24`, `sell_thr=90`).
  * Call `select_backend("cupy")`, pass backend to `make_buy_sell_masks`, and mirror the banner to `logs/phase08_masks_gpu.log`.
  * Keep `GPU_BUY_MASK` and `GPU_SELL_MASK` in scope for later phases.

**Notes**

* Initial run surfaced an interface mismatch (`'dict' object has no attribute 'asarray'`) due to passing the backend dict as `cp`. Fixed by allowing `make_buy_sell_masks` to accept either a CuPy module or the backend dict and extract the CuPy module internally.
* If `buy_thr >= sell_thr`, Phase 8 returns empty masks and a reason in `meta` without crashing.

**Result Snapshot**

* Banner prints once with:

  * `[PARAMS] buy_period=…, sell_period=…, buy_thr=…, sell_thr=…`
  * `[COUNTS] buy_ok=…, sell_ok=…`
  * `[SAMPLES] buy idx: … | sell idx: …` (up to 5 each)
  * `[REGIME] applied=None, true_count=<len(df)>` (unless a host regime mask is provided)
  * `[BACKEND] CuPy device=<name>, cc=<major.minor>`
* Log: `logs/phase08_masks_gpu.log`

**Verification**

* Confirm day 0 is False for both masks.
* Spot-check a few buy True dates on host:

  * `RSI_buy[d-1] < buy_thr`, `RSI_buy[d] < buy_thr`, and `Close[d] > Close[d-1]`.
* Spot-check a few sell True dates:

  * `RSI_sell[d-1] > sell_thr` and `RSI_sell[d] > sell_thr`.
* Confirm counts are plausible for `(buy_thr=24, sell_thr=90)` with RSI(2).

**Commit Message**


Phase 8: GPU predicate masks from precomputed RSI

- Add src/signals_gpu.py with make_buy_sell_masks()
- Two-day confirmation + buy momentum on GPU
- Optional regime mask intersection
- Add Phase-8 banner and file logging



## Phase 9 — MA\_GAP computation and regime labeling

**Goal**

* Compute `MA_GAP = ((MA_50 - MA_200) / max(MA_50, MA_200)) * 100` (row-wise, float64).
* Label each day into a configurable MA gap regime and expose a host boolean mask per regime for later phases.

**Changes**

* Added `src/regimes.py`:

  * `compute_ma_gap(df)` attaches `MA_GAP` (float64; no inf).
  * `generate_regime_labels(gap_ranges)` creates canonical labels, e.g., `gap_(None,-19)`.
  * `label_by_ranges(df, gap_ranges)` attaches `REGIME` by first-match into ranges with upper bound `< high`.
  * `regime_mask(df, label)` returns an all-True mask for `gap_all` or equality mask for a specific label.
* Added `src/banners/phase9.py` to print ranges, MA\_GAP coverage, counts, and first sample dates per regime.
* Updated `main.py` to mirror console to `logs/phase09_regimes.log` and keep:

  * `REGIME_LABELS` (ordered labels) and
  * `REGIME_MASK_HOST` (mask for a chosen label) for downstream use.

**Result Snapshot (SOXL run)**

* `[RANGES] (None,-19), (-19,7), (7,19), (19,31), (31,None)`
* `[MA_GAP] dtype=float64, non_null=3575 / total=3774, first_valid=2011-05-02`
* `[COUNTS] gap_(None,-19)=687, gap_(-19,7)=782, gap_(7,19)=1033, gap_(19,31)=927, gap_(31,None)=146`
* First 3 sample dates per regime printed for quick sanity.
* Log: `logs/phase09_regimes.log`

**Validation**

* `MA_GAP` aligned to `df.index`, float64, no ±inf.
* Every non-NaN `MA_GAP` row maps to exactly one regime via `< high` upper-bound rule.
* Handles degenerate `GAP_RANGES=[(None,None)]` as a single regime.



## Phase 10 — GPU Predicate Masks + Regime Split

Goal
- Combine Phase 8’s GPU predicate masks with Phase 9’s regime labels.
- Report buy/sell counts and sample dates per regime, and reconcile totals strictly within the MA_GAP domain (non-NaN rows).

Changes
- Added src/banners/phase10.py for a multi-section banner with per-regime counts/samples and integrity checks.
- Updated main.py Phase-10 block to:
  - Intersect device masks with each regime’s host mask (moved to device).
  - Reconcile totals against the MA_GAP domain (exclude warm-up NaNs).
  - Optionally report a diagnostic “gap_(NaN)” bucket for signals occurring in the NaN region.
  - Verify regimes are mutually exclusive and collectively exhaustive over non-NaN rows.
  - Audit boundary assignments for [low, high) edges using a small tolerance.
- Mirrored console to logs/phase10_masks_by_regime.log.

Result (SOXL, RSI(2), buy_thr=24, sell_thr=90)
- Totals (all rows): buy=56, sell=339
- Totals in MA_GAP domain: buy=53, sell=318
- Sum per regime: buy=53, sell=318  → matches domain totals
- NaN-domain diagnostics: buy=3, sell=21 (warm-up region)
- Overlaps=0, holes=0, boundaries sane
- Log: logs/phase10_masks_by_regime.log

Acceptance
- Per-regime counts sum exactly to domain totals.
- Samples are drawn only from in-domain rows where predicates are true.
- Integrity checks pass (no overlaps, no holes).

Commit Message
Phase 10: Masks by regime with domain reconciliation and integrity checks

- Intersect GPU masks with regime masks on device
- Reconcile to MA_GAP domain; add NaN diagnostics
- Overlap/hole checks and boundary audit
- Banner + logging to phase10_masks_by_regime.log
