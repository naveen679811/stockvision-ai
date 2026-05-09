"""
utils/data_handler.py
─────────────────────
Handles all data acquisition, cleaning, and feature engineering.
Supports yfinance (real-time) with optional Alpha Vantage fallback.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import logging
import ta
import os

logger = logging.getLogger(__name__)


# ── Popular stock presets ──────────────────────────────────────────────────────
POPULAR_STOCKS = {
    "NASDAQ": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX"],
    "NSE":    ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS",
               "BAJFINANCE.NS", "ICICIBANK.NS", "LT.NS"],
    "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"],
}

TIMEFRAME_MAP = {
    "1 Month":  ("1mo",  "1d"),
    "3 Months": ("3mo",  "1d"),
    "6 Months": ("6mo",  "1d"),
    "1 Year":   ("1y",   "1d"),
    "2 Years":  ("2y",   "1wk"),
    "5 Years":  ("5y",   "1wk"),
}


# ── Data Fetching ──────────────────────────────────────────────────────────────
def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Returns a clean DataFrame with:
      Open, High, Low, Close, Volume, Adj Close
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'.")

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        # Rename for uniformity
        df.columns = [c.title() for c in df.columns]
        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Data fetch failed for {ticker}: {e}")
        raise


def get_stock_info(ticker: str) -> dict:
    """Return company metadata (name, sector, market cap, PE, etc.)."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":        info.get("longName", ticker),
            "sector":      info.get("sector", "—"),
            "industry":    info.get("industry", "—"),
            "market_cap":  info.get("marketCap", 0),
            "pe_ratio":    info.get("trailingPE", None),
            "52w_high":    info.get("fiftyTwoWeekHigh", None),
            "52w_low":     info.get("fiftyTwoWeekLow", None),
            "avg_volume":  info.get("averageVolume", 0),
            "currency":    info.get("currency", "USD"),
            "description": info.get("longBusinessSummary", ""),
        }
    except Exception:
        return {"name": ticker, "sector": "—", "industry": "—",
                "market_cap": 0, "pe_ratio": None, "52w_high": None,
                "52w_low": None, "avg_volume": 0, "currency": "USD",
                "description": ""}


# ── Technical Indicators ───────────────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append all technical indicators using the `ta` library.
    Indicators added:
      SMA_20, SMA_50, EMA_12, EMA_26
      MACD, MACD_Signal, MACD_Hist
      RSI_14
      BB_Upper, BB_Middle, BB_Lower
      ATR_14, OBV, VWAP (approx)
    """
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # Moving Averages
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    df["EMA_12"] = ta.trend.ema_indicator(close, window=12)
    df["EMA_26"] = ta.trend.ema_indicator(close, window=26)

    # MACD
    macd = ta.trend.MACD(close)
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"]   = macd.macd_diff()

    # RSI
    df["RSI_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()
    df["BB_Width"]  = bb.bollinger_wband()

    # ATR
    df["ATR_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # OBV
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()

    # Approximate VWAP (daily)
    df["VWAP"] = (vol * (high + low + close) / 3).cumsum() / vol.cumsum()

    # Returns & volatility
    df["Daily_Return"]    = close.pct_change()
    df["Rolling_Std_20"]  = close.rolling(20).std()

    df.dropna(inplace=True)
    return df


# ── Preprocessing for LSTM ─────────────────────────────────────────────────────
def prepare_lstm_data(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "Close",
    window: int = 60,
    test_ratio: float = 0.2,
):
    """
    Prepare sliding-window sequences for LSTM training.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    scaler_features : fitted MinMaxScaler (features)
    scaler_target   : fitted MinMaxScaler (target only, for inverse transform)
    train_size      : int
    """
    data = df[feature_cols + [target_col]].copy().dropna()

    # Separate scalers so we can invert just the target
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target   = MinMaxScaler(feature_range=(0, 1))

    scaled_features = scaler_features.fit_transform(data[feature_cols])
    scaled_target   = scaler_target.fit_transform(data[[target_col]])

    # Combine
    scaled = np.hstack([scaled_features, scaled_target])
    n_features = len(feature_cols)

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i, :n_features])
        y.append(scaled[i, -1])

    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler_features, scaler_target, train_size


def inverse_transform_predictions(predictions: np.ndarray, scaler_target) -> np.ndarray:
    """Inverse-transform normalised predictions back to price scale."""
    return scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()


# ── Trading Signal Generation ──────────────────────────────────────────────────
def generate_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based signal engine combining RSI, MACD, and Bollinger Bands.

    Signal:  1 = BUY  |  0 = HOLD  |  -1 = SELL
    """
    df = df.copy()
    signals = pd.Series(0, index=df.index)

    rsi   = df.get("RSI_14")
    macd  = df.get("MACD")
    macd_s= df.get("MACD_Signal")
    close = df["Close"]
    bb_lo = df.get("BB_Lower")
    bb_hi = df.get("BB_Upper")

    if rsi is not None and macd is not None:
        buy_mask = (
            (rsi < 40) &
            (macd > macd_s) &
            (close <= bb_lo * 1.02)
        )
        sell_mask = (
            (rsi > 65) &
            (macd < macd_s) &
            (close >= bb_hi * 0.98)
        )
        signals[buy_mask]  =  1
        signals[sell_mask] = -1

    df["Signal"] = signals
    df["Signal_Label"] = signals.map({1: "BUY", 0: "HOLD", -1: "SELL"})
    return df


# ── Portfolio Utilities ────────────────────────────────────────────────────────
def calculate_portfolio_metrics(holdings: list[dict]) -> dict:
    """
    holdings: [{"ticker": "AAPL", "shares": 10, "avg_cost": 150.0}, ...]
    Returns portfolio-level P&L summary.
    """
    rows = []
    total_invested = 0
    total_current  = 0

    for h in holdings:
        try:
            info = yf.Ticker(h["ticker"]).fast_info
            current_price = info.last_price
        except Exception:
            current_price = h["avg_cost"]

        invested = h["shares"] * h["avg_cost"]
        current  = h["shares"] * current_price
        pnl      = current - invested
        pct      = (pnl / invested * 100) if invested else 0

        rows.append({
            "Ticker":        h["ticker"],
            "Shares":        h["shares"],
            "Avg Cost":      h["avg_cost"],
            "Current Price": round(current_price, 2),
            "Invested":      round(invested, 2),
            "Current Value": round(current, 2),
            "P&L":           round(pnl, 2),
            "P&L %":         round(pct, 2),
        })
        total_invested += invested
        total_current  += current

    return {
        "holdings_df": pd.DataFrame(rows),
        "total_invested": round(total_invested, 2),
        "total_current":  round(total_current, 2),
        "total_pnl":      round(total_current - total_invested, 2),
        "total_pnl_pct":  round((total_current - total_invested) / total_invested * 100
                                if total_invested else 0, 2),
    }
