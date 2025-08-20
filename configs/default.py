# configs/default.py
SETTINGS = {
    "LOOKBACK_BUFFER_DAYS": 30,
    "SOXL": {
        "ACTIVE": True,
        "START_DATE": "2010-01-01",
        "END_DATE": "2025-07-01",
        "INITIAL_CASH": 1000,
        "DAILY_CASH": 0,
        "BUY_THRESHOLD_MIN": 24,
        "BUY_THRESHOLD_MAX": 24,
        "SELL_THRESHOLD_MIN": 90,
        "SELL_THRESHOLD_MAX": 90,
        "RSI_MIN": 2,
        "RSI_MAX": 2,
        "RSI_WINDOW": 2,
        "RSI_PROXIMITY_PCT": 0.001,
        "CSV_CACHE_FILE": "SOXL_full_ohlc_indicators.csv",
        "TRANSACTION_MODE_ENABLED": False,
        "TRADE_HISTORY_BUY_RSI": 2,  # RSI period for buy signals in transaction history
        "TRADE_HISTORY_SELL_RSI": 2,  # RSI period for sell signals in transaction history
        "BUY_RSI_THRESHOLD": 22,  # intra 10 vs close 17 vs open 15
        "SELL_RSI_THRESHOLD": 72,  # intra 92 vs close 89 vs open 87
        "EXPENSE_RATIO": 0.0075,  # 0.75%
        "DISTRIBUTION_YIELD": 0.0217,  # 2.17%
        #"GAP_RANGES":[(None, -19), (-19, 7), (7, 19), (19, 31), (31, None)] 
        "GAP_RANGES":[(None, None)] 
    },
    "TQQQ": {
        "ACTIVE": True ,
        "START_DATE": "2012-01-01",
        "END_DATE": "2025-04-01",
        "INITIAL_CASH": 1000,
        "DAILY_CASH": 0,
        "BUY_THRESHOLD_MIN": 10,
        "BUY_THRESHOLD_MAX": 50,
        "SELL_THRESHOLD_MIN": 60,
        "SELL_THRESHOLD_MAX": 95,
        "RSI_MIN": 2,
        "RSI_MAX": 6,
        "RSI_WINDOW": 3,
        "RSI_PROXIMITY_PCT": 0.001,
        "CSV_CACHE_FILE": "TQQQ_full_ohlc_indicators.csv",
        "TRANSACTION_MODE_ENABLED": False,
        "TRADE_HISTORY_BUY_RSI": 3,  # RSI period for buy signals in transaction history
        "TRADE_HISTORY_SELL_RSI": 4,  # RSI period for sell signals in transaction history
        "BUY_RSI_THRESHOLD": 36,
        "SELL_RSI_THRESHOLD": 60,
        "EXPENSE_RATIO": 0.0084,  # 0.84%
        "DISTRIBUTION_YIELD": 0.0141,  # 1.41%
        "GAP_RANGES": [
        (None, -19),
        (-19, 7),
        (7, 19),
        (19, 31),
        (31, None),]
    },
    # RSI configurations for phase3 backtesting
    "RSI_PHASE3": {
        "SOXL": {
            "buy_periods": [2],
            "sell_periods": [2],
            "buy_thresholds": [22],
            "sell_thresholds": [70]
        },
        "TQQQ": {
            "buy_periods": [2],
            "sell_periods": [2],
            "buy_thresholds": [36],
            "sell_thresholds": [60]
        }
    },
    "QQQ": {
        "ACTIVE": False,
        "START_DATE": "2012-01-01",
        "END_DATE": "2025-04-01",
        "INITIAL_CASH": 0,
        "DAILY_CASH": 10,
        "BUY_THRESHOLD_MIN": 1,
        "BUY_THRESHOLD_MAX": 40,
        "SELL_THRESHOLD_MIN": 60,
        "SELL_THRESHOLD_MAX": 99,
        "RSI_MIN": 2,
        "RSI_MAX": 10,
        "RSI_WINDOW": 2,
        "RSI_PROXIMITY_PCT": 0.001,
        "CSV_CACHE_FILE": "QQQ_full_ohlc_indicators.csv",
        "TRANSACTION_MODE_ENABLED": False,
        "TRADE_HISTORY_BUY_RSI": 3,
        "TRADE_HISTORY_SELL_RSI": 2,
        "BUY_RSI_THRESHOLD": 30,
        "SELL_RSI_THRESHOLD": 98,
        "EXPENSE_RATIO": 0.0020,  # 0.20%
        "DISTRIBUTION_YIELD": 0.0058  # 0.58%
    },
    "QLD": {
        "ACTIVE": False,
        "START_DATE": "2015-01-01",
        "END_DATE": "2025-04-01",
        "INITIAL_CASH": 1000,
        "DAILY_CASH": 10,
        "BUY_THRESHOLD_MIN": 1,
        "BUY_THRESHOLD_MAX": 40,
        "SELL_THRESHOLD_MIN": 60,
        "SELL_THRESHOLD_MAX": 99,
        "RSI_MIN": 2,
        "RSI_MAX": 10,
        "RSI_WINDOW": 2,
        "RSI_PROXIMITY_PCT": 0.001,
        "CSV_CACHE_FILE": "QLD_full_ohlc_indicators.csv",
        "TRANSACTION_MODE_ENABLED": False,
        "TRADE_HISTORY_BUY_RSI": 3,
        "TRADE_HISTORY_SELL_RSI": 2,
        "BUY_RSI_THRESHOLD": 30,
        "SELL_RSI_THRESHOLD": 98,
        "EXPENSE_RATIO": 0.0095,  # 0.20%
        "DISTRIBUTION_YIELD": 0.0081  # 0.58%
    }
}


SYMBOLS = [
    symbol for symbol, config in SETTINGS.items()
    if isinstance(config, dict) and config.get("ACTIVE", False)
]