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