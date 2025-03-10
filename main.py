import io
import sys
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, request, render_template_string
from scipy.signal import argrelextrema  # For divergence detection

# Suppress logs/warnings
logging.getLogger("cmdstanpy").disabled = True
import warnings
warnings.filterwarnings("ignore", message="No frequency information was provided, so inferred frequency B will be used.")

# Constants
HORIZONS = [30, 60, 90, 180]

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def safe_float(value, default=None):
    if isinstance(value, str) and value.strip().upper() in ["", "NA", "N/A"]:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def format_value(value, fmt='.2f', scale=None):
    val = safe_float(value)
    if isinstance(val, float) and not pd.isna(val):
        if scale == 'T' and val >= 1e12:
            return f"{val / 1e12:{fmt}}T"
        elif scale == 'B' and val >= 1e9:
            return f"{val / 1e9:{fmt}}B"
        elif scale == 'M' and val >= 1e6:
            return f"{val / 1e6:{fmt}}M"
        elif scale == '%':
            return f"{val * 100:{fmt}}%"
        return f"{val:{fmt}}"
    return str(value)

def get_company_info(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.get_info()
    return info.get('longName', 'N/A'), info.get('longBusinessSummary', 'No summary available.')

def fetch_top_5_news(topic: str, api_key: str):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "apiKey": api_key,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("articles", [])[:5]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

def get_interval(chart_type: str) -> str:
    ct = chart_type.upper()
    mapping = {"1H": "60m", "1D": "1d", "1W": "1wk", "1M": "1mo"}
    return mapping.get(ct, "1d")

def fetch_data(ticker: str, start_date, end_date: datetime, chart_type: str) -> pd.DataFrame:
    interval = get_interval(chart_type)
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No data returned from Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Data after dropping NaNs is empty.")
    return df

def fetch_vix_data(start_date: datetime, end_date: datetime) -> pd.Series:
    vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d", progress=False)
    return vix['Close'] if not vix.empty else pd.Series(dtype=float)

# -------------------------------------------------------------------
# Divergence Detection Function
# -------------------------------------------------------------------
def detect_divergence(df: pd.DataFrame, window=5) -> (bool, bool):
    bullish_div = False
    bearish_div = False
    if 'Close' not in df.columns or 'RSI' not in df.columns or len(df) < window * 2:
        return bullish_div, bearish_div
    price_lows = argrelextrema(df['Close'].values, np.less, order=window)[0]
    if len(price_lows) >= 2:
        idx1, idx2 = price_lows[-2], price_lows[-1]
        if df['Close'].iloc[idx2] < df['Close'].iloc[idx1] and df['RSI'].iloc[idx2] > df['RSI'].iloc[idx1]:
            bullish_div = True
    price_highs = argrelextrema(df['Close'].values, np.greater, order=window)[0]
    if len(price_highs) >= 2:
        idx1, idx2 = price_highs[-2], price_highs[-1]
        if df['Close'].iloc[idx2] > df['Close'].iloc[idx1] and df['RSI'].iloc[idx2] < df['RSI'].iloc[idx1]:
            bearish_div = True
    return bullish_div, bearish_div

# -------------------------------------------------------------------
# Candlestick Pattern Detection Functions
# -------------------------------------------------------------------
def is_doji(open, high, low, close, body_threshold=0.1):
    body = abs(close - open)
    range_ = high - low
    return range_ > 0 and body < body_threshold * range_

def is_hammer(open, high, low, close):
    body = abs(close - open)
    upper_shadow = high - max(open, close)
    lower_shadow = min(open, close) - low
    return (body < 0.3 * (high - low)) and (lower_shadow > 2 * body) and (upper_shadow < 0.1 * body)

def is_hanging_man(open, high, low, close):
    body = abs(close - open)
    upper_shadow = high - max(open, close)
    lower_shadow = min(open, close) - low
    return (body < 0.3 * (high - low)) and (lower_shadow > 2 * body) and (upper_shadow < 0.1 * body)

def is_shooting_star(open, high, low, close):
    body = abs(close - open)
    upper_shadow = high - max(open, close)
    lower_shadow = min(open, close) - low
    return (body < 0.3 * (high - low)) and (upper_shadow > 2 * body) and (lower_shadow < 0.1 * body)

def is_inverted_hammer(open, high, low, close):
    body = abs(close - open)
    upper_shadow = high - max(open, close)
    lower_shadow = min(open, close) - low
    return (body < 0.3 * (high - low)) and (upper_shadow > 2 * body) and (lower_shadow < 0.1 * body)

def is_marubozu(open, high, low, close, shadow_threshold=0.05):
    body = abs(close - open)
    range_ = high - low
    upper_shadow = high - max(open, close)
    lower_shadow = min(open, close) - low
    return (upper_shadow < shadow_threshold * range_) and (lower_shadow < shadow_threshold * range_) and (body > 0.9 * range_)

def is_spinning_top(open, high, low, close):
    body = abs(close - open)
    upper_shadow = high - max(open, close)
    lower_shadow = min(open, close) - low
    range_ = high - low
    return (body < 0.3 * range_) and (upper_shadow > body) and (lower_shadow > body)

def is_bullish_engulfing(df, i):
    if i < 1:
        return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    return (prev['Close'] < prev['Open']) and (curr['Close'] > curr['Open']) and (curr['Open'] <= prev['Close']) and (curr['Close'] >= prev['Open'])

def is_bearish_engulfing(df, i):
    if i < 1:
        return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    return (prev['Close'] > prev['Open']) and (curr['Close'] < curr['Open']) and (curr['Open'] >= prev['Close']) and (curr['Close'] <= prev['Open'])

def is_bullish_harami(df, i):
    if i < 1:
        return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    return (prev['Close'] < prev['Open']) and (curr['Close'] > curr['Open']) and (curr['Open'] >= prev['Close']) and (curr['Close'] <= prev['Open'])

def is_bearish_harami(df, i):
    if i < 1:
        return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    return (prev['Close'] > prev['Open']) and (curr['Close'] < curr['Open']) and (curr['Open'] <= prev['Close']) and (curr['Close'] >= prev['Open'])

def is_morning_star(df, i):
    if i < 2:
        return False
    first = df.iloc[i-2]
    second = df.iloc[i-1]
    third = df.iloc[i]
    return (first['Close'] < first['Open']) and (abs(second['Close'] - second['Open']) < 0.3 * (second['High'] - second['Low'])) and (third['Close'] > third['Open']) and (third['Close'] > first['Open'])

def is_evening_star(df, i):
    if i < 2:
        return False
    first = df.iloc[i-2]
    second = df.iloc[i-1]
    third = df.iloc[i]
    return (first['Close'] > first['Open']) and (abs(second['Close'] - second['Open']) < 0.3 * (second['High'] - second['Low'])) and (third['Close'] < third['Open']) and (third['Close'] < first['Open'])

def detect_candlestick_patterns(df, lookback=5):
    patterns = []
    for i in range(-lookback, 0):
        idx = len(df) + i
        if idx < 0:
            continue
        candle = df.iloc[idx]
        date = df.index[idx]
        # Single Candle Patterns
        if is_doji(candle['Open'], candle['High'], candle['Low'], candle['Close']):
            patterns.append(f"Doji at {date}: Indecision in the market.")
        if is_hammer(candle['Open'], candle['High'], candle['Low'], candle['Close']) and candle['Close'] > candle['Open']:
            patterns.append(f"Hammer at {date}: Potential bullish reversal.")
        if is_hanging_man(candle['Open'], candle['High'], candle['Low'], candle['Close']) and candle['Close'] < candle['Open']:
            patterns.append(f"Hanging Man at {date}: Potential bearish reversal.")
        if is_shooting_star(candle['Open'], candle['High'], candle['Low'], candle['Close']) and candle['Close'] < candle['Open']:
            patterns.append(f"Shooting Star at {date}: Potential bearish reversal.")
        if is_inverted_hammer(candle['Open'], candle['High'], candle['Low'], candle['Close']) and candle['Close'] > candle['Open']:
            patterns.append(f"Inverted Hammer at {date}: Potential bullish reversal.")
        if is_marubozu(candle['Open'], candle['High'], candle['Low'], candle['Close']):
            direction = "Bullish" if candle['Close'] > candle['Open'] else "Bearish"
            patterns.append(f"{direction} Marubozu at {date}: Strong {direction.lower()} momentum.")
        if is_spinning_top(candle['Open'], candle['High'], candle['Low'], candle['Close']):
            patterns.append(f"Spinning Top at {date}: Indecision in the market.")
        # Two Candle Patterns
        if is_bullish_engulfing(df, idx):
            patterns.append(f"Bullish Engulfing at {date}: Potential upward reversal.")
        if is_bearish_engulfing(df, idx):
            patterns.append(f"Bearish Engulfing at {date}: Potential downward reversal.")
        if is_bullish_harami(df, idx):
            patterns.append(f"Bullish Harami at {date}: Potential bullish reversal.")
        if is_bearish_harami(df, idx):
            patterns.append(f"Bearish Harami at {date}: Potential bearish reversal.")
        # Three Candle Patterns
        if is_morning_star(df, idx):
            patterns.append(f"Morning Star at {date}: Potential bullish reversal.")
        if is_evening_star(df, idx):
            patterns.append(f"Evening Star at {date}: Potential bearish reversal.")
    return patterns

def analyze_candle_type(candle):
    open_price = candle['Open']
    close_price = candle['Close']
    high = candle['High']
    low = candle['Low']
    body = abs(close_price - open_price)
    range_ = high - low
    upper_shadow = high - max(open_price, close_price)
    lower_shadow = min(open_price, close_price) - low
    
    if range_ == 0:
        return "Invalid candle (no range)"
    
    direction = "Bullish" if close_price > open_price else "Bearish" if close_price < open_price else "Neutral"
    body_percent = body / range_ * 100
    upper_shadow_percent = upper_shadow / range_ * 100
    lower_shadow_percent = lower_shadow / range_ * 100
    
    characteristics = f"{direction} candle: Body {body_percent:.1f}% of range, Upper Shadow {upper_shadow_percent:.1f}%, Lower Shadow {lower_shadow_percent:.1f}%"
    return characteristics

# -------------------------------------------------------------------
# Scoring Functions
# -------------------------------------------------------------------
def compute_fundamental_score_original(fundamentals: dict) -> float:
    score = 0.0
    mc = safe_float(fundamentals.get("marketCap"), 0)
    score += 15 if mc >= 2e11 else 7.5
    trailing_pe = safe_float(fundamentals.get("trailingPE"), None)
    forward_pe = safe_float(fundamentals.get("forwardPE"), None)
    pe_subscore = 0
    if trailing_pe is not None:
        pe_subscore += 10 if trailing_pe < 15 else 0 if trailing_pe > 30 else 5
    if forward_pe is not None:
        pe_subscore += 10 if forward_pe < 15 else 0 if forward_pe > 30 else 5
    score += min(pe_subscore, 20)
    ps = safe_float(fundamentals.get("priceToSalesTrailing12Months"), None)
    if ps is not None:
        score += 5 if ps < 2 else 0 if ps > 10 else 2.5
    pb = safe_float(fundamentals.get("priceToBook"), None)
    if pb is not None:
        score += 5 if pb < 1 else 0 if pb > 5 else 2.5
    evrev = safe_float(fundamentals.get("enterpriseToRevenue"), None)
    if evrev is not None:
        score += 5 if evrev < 3 else 0 if evrev > 10 else 2.5
    evebitda = safe_float(fundamentals.get("enterpriseToEbitda"), None)
    if evebitda is not None:
        score += 5 if evebitda < 10 else 0 if evebitda > 20 else 2.5
    pm = safe_float(fundamentals.get("profitMargins"), None)
    if pm is not None:
        pm_percent = pm * 100
        score += 5 if pm_percent > 20 else 0 if pm_percent < 0 else 2.5
    roa = safe_float(fundamentals.get("returnOnAssets"), None)
    if roa is not None:
        roa_percent = roa * 100
        score += 5 if roa_percent > 10 else 0 if roa_percent < 2 else 2.5
    roe = safe_float(fundamentals.get("returnOnEquity"), None)
    if roe is not None:
        roe_percent = roe * 100
        score += 5 if roe_percent > 15 else 0 if roe_percent < 5 else 2.5
    net_inc = safe_float(fundamentals.get("netIncomeToCommon"), None)
    if net_inc is not None:
        score += 5 if net_inc > 0 else 0
    cash = safe_float(fundamentals.get("totalCash"), None)
    if cash is not None:
        score += 5 if cash >= 1e9 else 2.5
    dte = safe_float(fundamentals.get("debtToEquity"), None)
    if dte is not None:
        score += 10 if dte < 1 else 0 if dte > 2 else 5
    return score

def compute_technical_score_original(close_px, sma50, sma200, rsi_val, macd_val, macd_sig, bb_up, bb_lo, atr_val, stoch_k, open_px):
    score = 0.0
    close_px = safe_float(close_px, None)
    sma50 = safe_float(sma50, None)
    sma200 = safe_float(sma200, None)
    rsi_val = safe_float(rsi_val, None)
    macd_val = safe_float(macd_val, None)
    macd_sig = safe_float(macd_sig, None)
    bb_up = safe_float(bb_up, None)
    bb_lo = safe_float(bb_lo, None)
    atr_val = safe_float(atr_val, None)
    stoch_k = safe_float(stoch_k, None)
    open_px = safe_float(open_px, None)
    if close_px is not None and sma50 is not None and sma200 is not None:
        if close_px > sma50 and close_px > sma200:
            score += 20
    if rsi_val is not None:
        if rsi_val < 30:
            score += 20
        elif rsi_val < 70:
            score += 10
    if macd_val is not None and macd_sig is not None:
        if macd_val > macd_sig:
            score += 20
    if close_px is not None and bb_lo is not None and bb_up is not None:
        if close_px < bb_lo:
            score += 10
        elif close_px < bb_up:
            score += 5
    if atr_val is not None:
        if atr_val < 5:
            score += 10
        elif atr_val < 10:
            score += 5
    if stoch_k is not None:
        if stoch_k < 20:
            score += 10
        elif stoch_k < 80:
            score += 5
    if close_px is not None and open_px is not None and close_px > open_px:
        score += 10
    return score

def compute_fundamental_score_enhanced(fundamentals: dict, industry: str = None) -> float:
    score = 0.0
    weights = {'valuation': 0.4, 'profitability': 0.3, 'financial_health': 0.3}
    val_score = val_count = 0
    for key, threshold_high, threshold_low in [
        ("trailingPE", 15, 25),
        ("forwardPE", 15, 25),
        ("priceToSalesTrailing12Months", 2, 5),
        ("priceToBook", 1, 3)
    ]:
        val = safe_float(fundamentals.get(key), None)
        if val is not None:
            val_score += 20 if val < threshold_high else 5 if val < threshold_low else 0
            val_count += 1
    if val_count > 0:
        score += weights['valuation'] * (val_score / val_count) * 5
    prof_score = prof_count = 0
    pm = safe_float(fundamentals.get("profitMargins"), None)
    if pm is not None:
        pm_percent = pm * 100
        prof_score += 20 if pm_percent > 20 else 5 if pm_percent > 5 else 0
        prof_count += 1
    roa = safe_float(fundamentals.get("returnOnAssets"), None)
    if roa is not None:
        roa_percent = roa * 100
        prof_score += 20 if roa_percent > 10 else 5 if roa_percent > 3 else 0
        prof_count += 1
    roe = safe_float(fundamentals.get("returnOnEquity"), None)
    if roe is not None:
        roe_percent = roe * 100
        prof_score += 20 if roe_percent > 15 else 5 if roe_percent > 5 else 0
        prof_count += 1
    net_inc = safe_float(fundamentals.get("netIncomeToCommon"), None)
    if net_inc is not None:
        prof_score += 15 if net_inc > 1e9 else 5 if net_inc > 0 else 0
        prof_count += 1
    if prof_count > 0:
        score += weights['profitability'] * (prof_score / prof_count) * 5
    fin_score = fin_count = 0
    cash = safe_float(fundamentals.get("totalCash"), None)
    if cash is not None:
        fin_score += 15 if cash >= 5e9 else 5 if cash >= 1e9 else 0
        fin_count += 1
    dte = safe_float(fundamentals.get("debtToEquity"), None)
    if dte is not None:
        fin_score += 20 if dte < 0.5 else 5 if dte < 1.5 else 0
        fin_count += 1
    evrev = safe_float(fundamentals.get("enterpriseToRevenue"), None)
    if evrev is not None:
        fin_score += 15 if evrev < 2 else 5 if evrev < 5 else 0
        fin_count += 1
    evebitda = safe_float(fundamentals.get("enterpriseToEbitda"), None)
    if evebitda is not None:
        fin_score += 15 if evebitda < 8 else 5 if evebitda < 15 else 0
        fin_count += 1
    if fin_count > 0:
        score += weights['financial_health'] * (fin_score / fin_count) * 5
    return min(score, 100)

def compute_technical_score_enhanced(close_px, sma50, sma200, rsi_val, macd_val, macd_sig, bb_up, bb_lo, atr_val, stoch_k, open_px, vwap, ichimoku_span_a, ichimoku_span_b, adx_val, vix_val=None):
    score = 0.0
    weights = {'trend': 0.3, 'momentum': 0.3, 'volatility': 0.2, 'price_action': 0.2}
    close_px = safe_float(close_px, None)
    sma50 = safe_float(sma50, None)
    sma200 = safe_float(sma200, None)
    rsi_val = safe_float(rsi_val, None)
    macd_val = safe_float(macd_val, None)
    macd_sig = safe_float(macd_sig, None)
    bb_up = safe_float(bb_up, None)
    bb_lo = safe_float(bb_lo, None)
    atr_val = safe_float(atr_val, None)
    stoch_k = safe_float(stoch_k, None)
    open_px = safe_float(open_px, None)
    vwap = safe_float(vwap, None)
    ichimoku_span_a = safe_float(ichimoku_span_a, None)
    ichimoku_span_b = safe_float(ichimoku_span_b, None)
    adx_val = safe_float(adx_val, None)
    vix_val = safe_float(vix_val, None)
    trend_score = 0
    if close_px is not None and sma50 is not None and close_px > sma50:
        trend_score += 15
    if close_px is not None and sma200 is not None and close_px > sma200:
        trend_score += 15
    if ichimoku_span_a is not None and ichimoku_span_b is not None and close_px is not None and close_px > max(ichimoku_span_a, ichimoku_span_b):
        trend_score += 20
    score += weights['trend'] * trend_score
    mom_score = 0
    if rsi_val is not None:
        if rsi_val < 30:
            mom_score += 20
        elif rsi_val < 70:
            mom_score += 10
    if macd_val is not None and macd_sig is not None and macd_val > macd_sig:
        mom_score += 15
    if stoch_k is not None:
        if stoch_k < 20:
            mom_score += 15
        elif stoch_k < 80:
            mom_score += 5
    if adx_val is not None and adx_val > 25:
        mom_score += 15
    score += weights['momentum'] * mom_score
    vol_score = 0
    if close_px is not None and bb_lo is not None and bb_up is not None:
        if close_px < bb_lo:
            vol_score += 15
        elif close_px < bb_up:
            vol_score += 5
    if atr_val is not None:
        if atr_val < 3:
            vol_score += 15
        elif atr_val < 7:
            vol_score += 5
    score += weights['volatility'] * vol_score
    pa_score = 0
    if close_px is not None and vwap is not None and close_px > vwap:
        pa_score += 20
    if close_px is not None and open_px is not None and close_px > open_px:
        pa_score += 15
    score += weights['price_action'] * pa_score
    if vix_val is not None:
        if vix_val > 30:
            score *= 0.9
        elif vix_val < 20:
            score *= 1.1
    return min(max(score, 0), 100)

# -------------------------------------------------------------------
# Technical Indicator Functions
# -------------------------------------------------------------------
def rsi(series: pd.Series, period=14) -> pd.Series:
    return pd.Series(talib.RSI(series, timeperiod=period), index=series.index)

def ema(series: pd.Series, period=12) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def macd(price: pd.Series, short=12, long=26, signal=9):
    macd_line, signal_line, _ = talib.MACD(price, fastperiod=short, slowperiod=long, signalperiod=signal)
    return pd.Series(macd_line, index=price.index), pd.Series(signal_line, index=price.index)

def bollinger_bands(close: pd.Series, window=20, std_factor=2.0):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper_band = sma + std_factor * std
    lower_band = sma - std_factor * std
    return sma, upper_band, lower_band

def true_range(df: pd.DataFrame) -> pd.Series:
    shift_close = df['Close'].shift(1)
    h_l = df['High'] - df['Low']
    h_pc = (df['High'] - shift_close).abs()
    l_pc = (df['Low'] - shift_close).abs()
    return pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    return pd.Series(talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period), index=df.index)

def stochastic_oscillator(df: pd.DataFrame, k_period=14, d_period=3):
    stoch_k, stoch_d = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=k_period, slowk_period=3, slowd_period=d_period)
    return pd.Series(stoch_k, index=df.index), pd.Series(stoch_d, index=df.index)

def vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

def ichimoku_cloud(df: pd.DataFrame):
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    tenkan_sen = (high_9 + low_9) / 2
    kijun_sen = (high_26 + low_26) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)
    chikou_span = df['Close'].shift(-26)
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def adx(df: pd.DataFrame, period=14) -> pd.Series:
    return pd.Series(talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=period), index=df.index)

# -------------------------------------------------------------------
# Pattern Detection
# -------------------------------------------------------------------
def detect_rectangle(df: pd.DataFrame, lookback=30, tolerance=0.02) -> bool:
    if len(df) < lookback:
        return False
    subset = df.tail(lookback)
    max_p = float(subset['High'].max())
    min_p = float(subset['Low'].min())
    avg_p = float(subset['Close'].mean())
    return ((max_p - min_p) / avg_p) < tolerance

def detect_double_top_bottom(df: pd.DataFrame, lookback=50, tolerance=0.02) -> str:
    if len(df) < lookback:
        return 'none'
    closes = df['Close'].tail(lookback).values
    peak1_idx = np.argmax(closes)
    peak1 = closes[peak1_idx]
    if peak1_idx < len(closes) - 1:
        peak2_idx = peak1_idx + np.argmax(closes[peak1_idx + 1:]) + 1
        peak2 = closes[peak2_idx]
        if abs(peak2 - peak1) / peak1 <= tolerance and peak2_idx != peak1_idx:
            return 'double_top'
    bottom1_idx = np.argmin(closes)
    bottom1 = closes[bottom1_idx]
    if bottom1_idx < len(closes) - 1:
        bottom2_idx = bottom1_idx + np.argmin(closes[bottom1_idx + 1:]) + 1
        bottom2 = closes[bottom2_idx]
        if abs(bottom2 - bottom1) / bottom1 <= tolerance and bottom2_idx != bottom1_idx:
            return 'double_bottom'
    return 'none'

def detect_breakaway_gap(df: pd.DataFrame, gap_factor=1.02) -> bool:
    if len(df) < 2:
        return False
    today = df.iloc[-1]
    yday = df.iloc[-2]
    return float(today['Open']) > (float(yday['High']) * gap_factor)

def build_final_summary(fund_signals: dict, tech_outlook: str) -> str:
    bullish_fund = fund_signals["bullish"]
    bearish_fund = fund_signals["bearish"]
    if tech_outlook == "Likely Uptrend" and bullish_fund and not bearish_fund:
        return "Overall Summary: Strong Bullish Confluence (fundamentals + uptrend)."
    elif tech_outlook == "Likely Downtrend" and bearish_fund and not bullish_fund:
        return "Overall Summary: Strong Bearish Confluence (fundamentals + downtrend)."
    else:
        return "Overall Summary: Mixed or Inconclusive. Further analysis advised."

def show_quarterly_performance(df: pd.DataFrame, last_n=6) -> None:
    quarterly_close = df['Close'].resample('QE').last()
    pct_change = quarterly_close.pct_change() * 100
    quarterly_perf = pd.DataFrame({
        'Quarter End': quarterly_close.index,
        'Close': quarterly_close.values,
        'QoQ % Change': pct_change.values
    }).dropna().tail(last_n)
    print(f"\n=== Last {last_n} Quarters Performance ===")
    print(quarterly_perf.to_string(index=False, formatters={'Close': '{:.2f}'.format, 'QoQ % Change': '{:.2f}'.format}))

def add_fibonacci_retracement(fig, df: pd.DataFrame, lookback: int = 100):
    if len(df) < lookback:
        return
    sub = df.tail(lookback)
    swing_high = sub['High'].max()
    swing_low = sub['Low'].min()
    diff = swing_high - swing_low
    levels = {
        '0%': swing_high,
        '23.6%': swing_high - diff * 0.236,
        '38.2%': swing_high - diff * 0.382,
        '50%': swing_high - diff * 0.5,
        '61.8%': swing_high - diff * 0.618,
        '100%': swing_low
    }
    for fib_label, fib_price in levels.items():
        fig.add_hline(y=fib_price, line_width=1, line_dash='dash', line_color='gray', row=1, col=1)

def print_detailed_fundamentals(ticker: str, fundamentals: dict) -> dict:
    print(f"\n=== Fundamental Analysis ({ticker}) ===")
    signals = {"bullish": False, "bearish": False}
    long_name = fundamentals.get("longName", "N/A")
    print(f"longName: {long_name}")
    keys = [
        ("recommendationKey", fundamentals.get("recommendationKey")),
        ("Market Cap", fundamentals.get("marketCap")),
        ("Enterprise Value", fundamentals.get("enterpriseValue")),
        ("Trailing P/E", fundamentals.get("trailingPE")),
        ("Forward P/E", fundamentals.get("forwardPE")),
        ("Price/Sales (ttm)", fundamentals.get("priceToSalesTrailing12Months")),
        ("Price/Book (mrq)", fundamentals.get("priceToBook")),
        ("EV/Revenue", fundamentals.get("enterpriseToRevenue")),
        ("EV/EBITDA", fundamentals.get("enterpriseToEbitda")),
        ("Profit Margin", fundamentals.get("profitMargins")),
        ("Return on Assets (ttm)", fundamentals.get("returnOnAssets")),
        ("Return on Equity (ttm)", fundamentals.get("returnOnEquity")),
        ("Revenue (ttm)", fundamentals.get("totalRevenue")),
        ("Net Income Avl to Common (ttm)", fundamentals.get("netIncomeToCommon")),
        ("Diluted EPS (ttm)", fundamentals.get("trailingEps")),
        ("Total Cash (mrq)", fundamentals.get("totalCash")),
        ("Total Debt/Equity (mrq)", fundamentals.get("debtToEquity")),
        ("Levered Free Cash Flow (ttm)", fundamentals.get("freeCashflow"))
    ]
    for key, value in keys:
        comment = ""
        if value is not None:
            if key in ["Market Cap", "Enterprise Value", "Revenue (ttm)", "Net Income Avl to Common (ttm)", "Total Cash (mrq)", "Levered Free Cash Flow (ttm)"]:
                val_str = format_value(value, scale='T' if value >= 1e12 else 'B' if value >= 1e9 else 'M')
            elif key in ["Profit Margin", "Return on Assets (ttm)", "Return on Equity (ttm)"]:
                val_str = format_value(value, scale='%')
            else:
                val_str = format_value(value)
            if key == "recommendationKey":
                lower_val = str(value).lower()
                if lower_val in ("buy", "strong_buy"):
                    comment = "Bullish"
                    signals["bullish"] = True
                elif lower_val in ("sell", "strong_sell"):
                    comment = "Bearish"
                    signals["bearish"] = True
                elif lower_val == "hold":
                    comment = "Neutral"
                else:
                    comment = "Neutral"
            elif key == "Trailing P/E":
                val = safe_float(value, default=None)
                if val is not None:
                    if val < 15:
                        comment = "Positive"
                        signals["bullish"] = True
                    elif val > 30:
                        comment = "Negative"
                        signals["bearish"] = True
                    else:
                        comment = "Neutral"
            elif key == "Forward P/E":
                val = safe_float(value, default=None)
                if val is not None:
                    if val < 15:
                        comment = "Positive"
                        signals["bullish"] = True
                    elif val > 30:
                        comment = "Negative"
                        signals["bearish"] = True
                    else:
                        comment = "Neutral"
            if comment:
                print(f"{key}: {val_str} → {comment}")
            else:
                print(f"{key}: {val_str}")
        else:
            print(f"{key}: N/A")
    return signals

def generate_chart_html(df: pd.DataFrame, ticker: str, chart_type: str) -> str:
    sr_lookback = 250
    sr_subset = df.tail(sr_lookback)
    support_val = float(sr_subset['Low'].min())
    resistance_val = float(sr_subset['High'].max())
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.2, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#00FF00', decreasing_line_color='#FF0000'), row=1, col=1)
    for col, color, name in [
        ('SMA50', 'cyan', 'SMA50'),
        ('SMA200', 'orange', 'SMA200'),
        ('EMA20', 'purple', 'EMA20'),
        ('EMA50', 'pink', 'EMA50'),
        ('VWAP', 'yellow', 'VWAP')
    ]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1), name=name), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(color='green', width=1), name='Span A'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SpanB'], line=dict(color='red', width=1), name='Span B'), row=1, col=1)
    fig.add_hline(y=support_val, line_dash='dash', line_color="green", annotation_text="Support", row=1, col=1)
    fig.add_hline(y=resistance_val, line_dash='dash', line_color="red", annotation_text="Resistance", row=1, col=1)
    add_fibonacci_retracement(fig, df, lookback=100)
    colors_volume = ['green' if c > o else 'red' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors_volume, opacity=0.7), row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='cyan', width=2), name='RSI'), row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.2, line_width=0, row=3, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='white', width=1), name='MACD'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='yellow', width=1), name='MACD_Signal'), row=4, col=1)
    macd_hist_colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=macd_hist_colors, name='MACD_Hist'), row=4, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_layout(
        title=f"{ticker} - {chart_type} Analysis",
        template=None, paper_bgcolor="black", plot_bgcolor="black",
        font=dict(color="white"),
        width=1100, height=900, xaxis_rangeslider_visible=False
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def run_analysis(ticker, start_date, end_date, chart_type):
    df = fetch_data(ticker, start_date, end_date, chart_type)
    freq = pd.infer_freq(df.index) or 'B'
    df = df.asfreq(freq, method='ffill')
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

# -------------------------------------------------------------------
# Updated run_all_web Function
# -------------------------------------------------------------------
def run_all_web(ticker, start_date, end_date, chart_type, horizons):
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            df = run_analysis(ticker, start_date, end_date, chart_type)
            df_analytics = df.copy()
            analyze_data(df_analytics, ticker, chart_type, start_date, end_date)
            df_chart = df_analytics.tail(250).copy()
            chart_html = generate_chart_html(df_chart, ticker, chart_type)
        except ValueError as e:
            print(f"Error: {e}")
            return str(e), ""
    return buf.getvalue(), chart_html

def analyze_data(df: pd.DataFrame, ticker: str, chart_type: str, start_date, end_date: datetime):
    show_quarterly_performance(df, last_n=6)
    # Technical Indicators
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['RSI'] = rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'] = macd(df['Close'])
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['BB_Mid'], df['BB_Upper'], df['BB_Lower'] = bollinger_bands(df['Close'], 20, 2)
    df['ATR'] = atr(df, 14)
    df['%K'], df['%D'] = stochastic_oscillator(df, 14, 3)
    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)
    df['VWAP'] = vwap(df)
    df['Tenkan'], df['Kijun'], df['SpanA'], df['SpanB'], df['Chikou'] = ichimoku_cloud(df)
    df['ADX'] = adx(df, 14)
    df.dropna(subset=['Close', 'SMA50', 'SMA200', 'RSI', 'MACD', 'MACD_Signal', 'BB_Mid', 'ATR', '%K', 'EMA20', 'VWAP'], inplace=True)
    if df.empty:
        raise ValueError("Not enough data after rolling calculations.")
    vix_data = fetch_vix_data(start_date, end_date)
    latest_vix = vix_data.iloc[-1] if not vix_data.empty else None

    fundamentals = yf.Ticker(ticker).get_info()
    industry = fundamentals.get('industry', None)
    fund_signals = print_detailed_fundamentals(ticker, fundamentals)
    fund_score_original = compute_fundamental_score_original(fundamentals)
    fund_score_enhanced = compute_fundamental_score_enhanced(fundamentals, industry)
    print(f"\nOriginal Fundamental Score (0-100): {format_value(fund_score_original)}")
    print(f"Enhanced Fundamental Score (0-100): {format_value(fund_score_enhanced)}")
    latest = df.iloc[-1]
    close_px = latest['Close']
    sma50_val = latest['SMA50']
    sma200_val = latest['SMA200']
    rsi_val = latest['RSI']
    macd_val = latest['MACD']
    macd_sig = latest['MACD_Signal']
    bb_up = latest['BB_Upper']
    bb_lo = latest['BB_Lower']
    atr_val = latest['ATR']
    k_val = latest['%K']
    d_val = latest['%D']
    ema20_val = latest['EMA20']
    ema50_val = latest['EMA50']
    vwap_val = latest['VWAP']
    span_a_val = latest['SpanA'] if not pd.isna(latest['SpanA']) else None
    span_b_val = latest['SpanB'] if not pd.isna(latest['SpanB']) else None
    adx_val = latest['ADX'] if not pd.isna(latest['ADX']) else None
    print(f"\n--- {chart_type} Chart Analysis ({ticker}) ---")
    print(f"Last Bar Date: {df.index[-1]}")
    print(f"Close Price: {format_value(close_px)}")
    sma50_comment = "Positive (above SMA50)" if close_px > sma50_val else "Negative (below SMA50)"
    sma200_comment = "Positive (above SMA200)" if close_px > sma200_val else "Negative (below SMA200)"
    ema20_comment = "Positive (above EMA20)" if close_px > ema20_val else "Negative (below EMA20)"
    ema50_comment = "Positive (above EMA50)" if close_px > ema50_val else "Negative (below EMA50)"
    rsi_comment = "Neutral"
    if rsi_val < 30:
        rsi_comment = "Positive (RSI <30 => Oversold)"
    elif rsi_val > 70:
        rsi_comment = "Negative (RSI >70 => Overbought)"
    macd_comment = "Neutral"
    if macd_val > macd_sig:
        macd_comment = "Positive (MACD > Signal)"
    elif macd_val < macd_sig:
        macd_comment = "Negative (MACD < Signal)"
    bb_comment = "Neutral"
    if close_px > bb_up:
        bb_comment = "Negative (above upper band)"
    elif close_px < bb_lo:
        bb_comment = "Positive (below lower band)"
    atr_comment = "Negative (high volatility)" if atr_val > 5 else "Neutral"
    stoch_comment = "Neutral"
    if k_val < 20:
        stoch_comment = "Positive (%K < 20 => Oversold)"
    elif k_val > 80:
        stoch_comment = "Negative (%K > 80 => Overbought)"
    vwap_comment = "Positive (above VWAP)" if close_px > vwap_val else "Negative (below VWAP)"
    ichimoku_comment = "Neutral"
    if span_a_val is not None and span_b_val is not None:
        if close_px > max(span_a_val, span_b_val):
            ichimoku_comment = "Positive (above cloud)"
        elif close_px < min(span_a_val, span_b_val):
            ichimoku_comment = "Negative (below cloud)"
    adx_comment = "Positive (ADX > 25 => Strong trend)" if adx_val > 25 else "Neutral"
    print(f"SMA50 = {format_value(sma50_val)} → {sma50_comment}")
    print(f"SMA200 = {format_value(sma200_val)} → {sma200_comment}")
    print(f"EMA20 = {format_value(ema20_val)} → {ema20_comment}")
    print(f"EMA50 = {format_value(ema50_val)} → {ema50_comment}")
    print(f"RSI = {format_value(rsi_val)} → {rsi_comment}")
    print(f"MACD = {format_value(macd_val)}, Signal = {format_value(macd_sig)} → {macd_comment}")
    print(f"BollingerUp = {format_value(bb_up)}, BollingerLow = {format_value(bb_lo)} → {bb_comment}")
    print(f"ATR(14) = {format_value(atr_val)} → {atr_comment}")
    print(f"Stoch %K = {format_value(k_val)}, %D = {format_value(d_val)} → {stoch_comment}")
    print(f"VWAP = {format_value(vwap_val)} → {vwap_comment}")
    print(f"Ichimoku SpanA = {format_value(span_a_val) if span_a_val else 'N/A'}, SpanB = {format_value(span_b_val) if span_b_val else 'N/A'} → {ichimoku_comment}")
    print(f"ADX = {format_value(adx_val) if adx_val else 'N/A'} → {adx_comment}")
    sr_lookback = 100
    sr_subset = df.tail(sr_lookback)
    support_val = float(sr_subset['Low'].min())
    resistance_val = float(sr_subset['High'].max())
    print(f"\nSupport/Resistance (last {sr_lookback} bars):")
    print(f"  Nearest support ~{format_value(support_val)}")
    print(f"  Nearest resistance ~{format_value(resistance_val)}")
    above_sma = close_px > sma50_val and close_px > sma200_val
    macd_bullish = macd_val > macd_sig
    rsi_normal = rsi_val < 70
    below_sma = close_px < sma50_val and close_px < sma200_val
    macd_bearish = macd_val < macd_sig
    rsi_not_oversold = rsi_val > 30
    tech_outlook = ("Likely Uptrend" if above_sma and macd_bullish and rsi_normal
                    else "Likely Downtrend" if below_sma and macd_bearish and rsi_not_oversold
                    else "Mixed Signals")
    print(f"\nOverall Technical Outlook: {tech_outlook}")
    patterns_found = []
    if detect_rectangle(df, 30, 0.02):
        patterns_found.append("Rectangle")
    dtb = detect_double_top_bottom(df, 50, 0.02)
    if dtb == 'double_top':
        patterns_found.append("Double Top")
    elif dtb == 'double_bottom':
        patterns_found.append("Double Bottom")
    if detect_breakaway_gap(df, 1.02):
        patterns_found.append("Breakaway Gap")
    bullish_div, bearish_div = detect_divergence(df, window=5)
    if bullish_div:
        patterns_found.append("Bullish Divergence")
    if bearish_div:
        patterns_found.append("Bearish Divergence")
    print("\nDetected Pattern(s):" if patterns_found else "\nNo major pattern detected.")
    for pat in patterns_found:
        print(" -", pat)
    tech_score_original = compute_technical_score_original(close_px, sma50_val, sma200_val, rsi_val, macd_val, macd_sig, bb_up, bb_lo, atr_val, k_val, latest['Open'])
    tech_score_enhanced = compute_technical_score_enhanced(close_px, sma50_val, sma200_val, rsi_val, macd_val, macd_sig, bb_up, bb_lo, atr_val, k_val, latest['Open'], vwap_val, span_a_val, span_b_val, adx_val, latest_vix)
    print(f"\nOriginal Technical Score (0-100): {format_value(tech_score_original)}")
    print(f"Enhanced Technical Score (0-100): {format_value(tech_score_enhanced)}")
    summary_final = build_final_summary(fund_signals, tech_outlook)
    print(f"\n{summary_final}")
    if len(df) >= 100:
        fib_sub = df.tail(100)
        swing_high = fib_sub['High'].max()
        swing_low = fib_sub['Low'].min()
        diff = swing_high - swing_low
        fib_levels = {
            '0%': swing_high,
            '23.6%': swing_high - diff * 0.236,
            '38.2%': swing_high - diff * 0.382,
            '50%': swing_high - diff * 0.5,
            '61.8%': swing_high - diff * 0.618,
            '100%': swing_low
        }
        print("\nFibonacci Retracement Levels (last 100 bars):")
        for fib_label, fib_price in fib_levels.items():
            print(f"  {fib_label}: {format_value(fib_price)}")

    # Candlestick Pattern Analysis
    print("\n=== Candlestick Pattern Analysis ===")
    print("Latest Candle Analysis:")
    latest_candle = df.iloc[-1]
    candle_type = analyze_candle_type(latest_candle)
    print(f" - {candle_type}")
    candle_patterns = detect_candlestick_patterns(df, lookback=5)
    if candle_patterns:
        print("Detected Candlestick Patterns in the last 5 candles:")
        for pattern in candle_patterns:
            print(f" - {pattern}")
    else:
        print("No significant candlestick patterns detected in the last 5 candles.")
    # Summary of Candlestick Patterns
    bullish_count = sum(1 for p in candle_patterns if "bullish" in p.lower())
    bearish_count = sum(1 for p in candle_patterns if "bearish" in p.lower())
    neutral_count = len(candle_patterns) - bullish_count - bearish_count
    print("\nCandlestick Pattern Summary:")
    print(f" - Bullish Patterns: {bullish_count}")
    print(f" - Bearish Patterns: {bearish_count}")
    print(f" - Neutral Patterns: {neutral_count}")
    sentiment = "Bullish" if bullish_count > bearish_count else "Bearish" if bearish_count > bullish_count else "Neutral"
    print(f" - Overall Sentiment: {sentiment}")

# -------------------------------------------------------------------
# Flask App
# -------------------------------------------------------------------
app = Flask(__name__)
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Stock Analysis</title>
    <style>
        body { background-color: #121212; color: #e0e0e0; font-family: Arial, sans-serif; }
        pre { background-color: #1e1e1e; padding: 1em; border-radius: 5px; overflow-x: auto; }
        .container { width: 90%; margin: auto; }
        .form-row { margin-bottom: 1em; }
        input[type="text"] { padding: 0.5em; width: 300px; }
        select { padding: 0.5em; }
        input[type="submit"] { padding: 0.5em 1em; }
        h1, h2 { color: #bdbdbd; }
    </style>
</head>
<body>
<div class="container">
    <h1>Stock Analysis</h1>
    <form method="post">
        <div class="form-row">
            <label for="ticker">Enter Ticker Symbol (e.g., TSLA, AAPL, NVDA):</label>
            <input type="text" id="ticker" name="ticker" value="{{ ticker|default('TSLA') }}">
        </div>
        <div class="form-row">
            <label for="chart_type">Select Chart Type:</label>
            <select id="chart_type" name="chart_type">
                <option value="1H" {% if chart_type == '1H' %} selected {% endif %}>1 Hour</option>
                <option value="1D" {% if chart_type == '1D' %} selected {% endif %}>1 Day</option>
                <option value="1W" {% if chart_type == '1W' %} selected {% endif %}>1 Week</option>
                <option value="1M" {% if chart_type == '1M' %} selected {% endif %}>1 Month</option>
            </select>
        </div>
        <input type="submit" value="Analyze">
    </form>
    {% if analysis %}
        <h2>Analysis Output</h2>
        <pre>{{ analysis }}</pre>
    {% endif %}
    {% if chart %}
        <h2>Chart</h2>
        {{ chart|safe }}
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    analysis = ""
    chart = ""
    ticker = "TSLA"
    chart_type = "1D"
    if request.method == "POST":
        ticker = request.form.get("ticker", "TSLA").upper()
        chart_type = request.form.get("chart_type", "1D").upper()
        if chart_type == "1D":
            START_DATE = datetime(1900, 1, 1)
        else:
            months_back = {'1H': 18, '4H': 30, '1D': 60, '1W': 250, '1M': 300}.get(chart_type, 60)
            END_DATE = datetime.today()
            START_DATE = END_DATE - relativedelta(months=months_back)
        END_DATE = datetime.today()
        analysis, chart = run_all_web(ticker, START_DATE, END_DATE, chart_type, HORIZONS)
    return render_template_string(template, analysis=analysis, chart=chart, ticker=ticker, chart_type=chart_type)

if __name__ == '__main__':
    app.run(debug=True, port=5015)
