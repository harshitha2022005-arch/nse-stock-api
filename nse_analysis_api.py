"""
NSE Stock Analysis API  (version 2)
================================================================================
FastAPI backend for the "NSE Terminal" dashboard. Data comes free from Yahoo
Finance (yfinance); every indicator is calculated here with pandas / numpy.

ENDPOINTS
---------
  Version 1 (unchanged):
    GET /analyze/{symbol}            full technical + fundamental analysis
    GET /chart/{symbol}?months=6     candlestick data + RSI, MACD, MA50, MA200
    GET /watchlist?symbols=A,B,C     quick snapshot of up to 10 stocks

  Version 2 (new):
    GET /market-overview             NIFTY 50, SENSEX, BANK NIFTY, NIFTY IT
    GET /signals/{symbol}            7-indicator bullish/bearish scorecard
    GET /risk/{symbol}?years=2       volatility, Sharpe, drawdown, VaR, beta
    GET /compare?symbols=A,B&months=12        stocks rebased to 100
    GET /correlation?symbols=A,B,C&months=12  correlation matrix
    GET /portfolio?symbols=A,B&weights=60,40  portfolio risk / return
    GET /backtest/{symbol}?strategy=ma_crossover   strategy vs buy-and-hold

REQUIREMENTS (requirements.txt - unchanged)
    fastapi, uvicorn, yfinance, pandas, numpy

RUN LOCALLY
    uvicorn nse_analysis_api:app --reload --port 8000
    then open http://127.0.0.1:8000/docs  (auto-generated test page)

RENDER START COMMAND
    uvicorn nse_analysis_api:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

app = FastAPI(title="NSE Stock Analysis API", version="2.0")

# Allow any frontend (v0.dev, Lovable, or anywhere else) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SHARED HELPERS (new in v2) ────────────────────────────────────────────

# Simple in-memory cache: the same stock is not re-downloaded from Yahoo for
# 10 minutes. Makes the app faster and reduces the chance of Yahoo blocking us.
_CACHE = {}
CACHE_TTL_SECONDS = 600

# Approximate Indian risk-free rate (10-year government bond yield, ~6.5%).
# Used in Sharpe / Sortino ratios. Change it here if you want a different assumption.
RISK_FREE_RATE = 0.065


def to_ticker(symbol: str) -> str:
    """RELIANCE -> RELIANCE.NS ; index symbols like ^NSEI are left as they are."""
    symbol = symbol.upper().strip()
    if symbol.startswith("^") or symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"


def short_name(ticker: str) -> str:
    return ticker.replace(".NS", "")


def parse_symbols(symbols: str, min_n: int, max_n: int):
    """Turn 'tcs, infy,TCS' into ['TCS', 'INFY'] (upper-case, no duplicates) and validate the count."""
    cleaned = []
    for s in symbols.split(","):
        s = s.strip().upper()
        if s and s not in cleaned:
            cleaned.append(s)
    if len(cleaned) < min_n:
        raise HTTPException(status_code=400, detail=f"Provide at least {min_n} different symbol(s), comma-separated.")
    if len(cleaned) > max_n:
        raise HTTPException(status_code=400, detail=f"Maximum {max_n} symbols allowed for this endpoint.")
    return cleaned


def get_history(ticker: str, period: str = "2y"):
    """Download daily price history (cached). Index is cleaned to plain dates."""
    key = (ticker, period)
    now = time.time()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1].copy()

    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if hist is not None and not hist.empty:
        idx = hist.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        hist.index = idx.normalize()
        if len(_CACHE) > 100:
            _CACHE.clear()
        _CACHE[key] = (now, hist)
        return hist.copy()
    return hist


def period_for_years(years: float) -> str:
    """Yahoo only accepts fixed period strings (1y, 2y, 5y, 10y) — pick the smallest that fits."""
    if years <= 1:
        return "1y"
    if years <= 2:
        return "2y"
    if years <= 5:
        return "5y"
    return "10y"


def load_closes(tickers, trading_days: int):
    """Closing prices of several stocks side by side, only on days when ALL of them traded."""
    period = period_for_years(trading_days / 252)
    frames = {}
    for t in tickers:
        h = get_history(t, period)
        if h is None or h.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {t}. Check the symbol is correct.")
        frames[short_name(t)] = h["Close"]
    df = pd.concat(frames, axis=1).dropna().tail(trading_days)
    if len(df) < 30:
        raise HTTPException(status_code=400, detail="Not enough overlapping price history for these symbols.")
    return df


def perf_stats(returns, rf: float = RISK_FREE_RATE):
    """Standard performance numbers from a series of daily returns."""
    returns = returns.dropna()
    n = len(returns)
    if n < 2:
        return {}
    growth = (1 + returns).cumprod()
    total = growth.iloc[-1] - 1
    cagr = growth.iloc[-1] ** (252 / n) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol and vol > 0 else None
    drawdown = growth / growth.cummax() - 1
    return {
        "total_return_pct": safe_round(total * 100),
        "cagr_pct": safe_round(cagr * 100),
        "volatility_pct": safe_round(vol * 100),
        "sharpe_ratio": safe_round(sharpe),
        "max_drawdown_pct": safe_round(drawdown.min() * 100),
    }


def downsample(df, max_points: int = 300):
    """Keep chart payloads small: at most ~max_points rows, always including the last row."""
    step = max(1, len(df) // max_points)
    out = df.iloc[::step]
    if out.index[-1] != df.index[-1]:
        out = pd.concat([out, df.iloc[[-1]]])
    return out

# ── INDICATOR FUNCTIONS (same ones built in our Jupyter session today) ────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_moving_averages(prices, short_window=50, long_window=200):
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()
    return ma_short, ma_long

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100

def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth — the corrected version, fixing the bug
    we caught earlier today where fixed-position matching broke on
    stocks with an irregularly reported quarter."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def safe_round(val, decimals=2):
    if val is None or pd.isna(val) or not np.isfinite(val):
        return None
    return round(float(val), decimals)


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    """Same date-matched logic as earnings growth, applied to revenue."""
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Extended fundamentals: earnings growth, revenue growth, margins, ROE, debt-to-equity.

    IMPORTANT: this deliberately avoids stock.info — Yahoo Finance blocks/rate-limits
    that specific endpoint for many cloud hosting providers (Render, Railway, Heroku),
    returning an almost-empty dict even though the price-history endpoint works fine.
    Instead, everything here is calculated directly from the income statement and
    balance sheet, which use the endpoint that's actually working for us."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        # margins calculated directly — same math a real analyst would do by hand
        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)

    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None

            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    equity_series = balance_sheet.loc[equity_label].dropna()
                    if len(equity_series) > 0:
                        equity = equity_series.iloc[0]
                        break

            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    debt_series = balance_sheet.loc[debt_label].dropna()
                    if len(debt_series) > 0:
                        total_debt = debt_series.iloc[0]
                        break

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                # ROE using latest quarter's net income annualized (x4) against latest equity —
                # a simplified TTM-style approximation, standard when only one quarter is available
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)

    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    # market cap: try the lighter fast_info endpoint, which is less likely to be blocked
    # than the full .info dict — separate try block so a failure here doesn't affect anything above
    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass  # market cap is a nice-to-have; not worth failing the request over

    return fundamentals


def get_technical_snapshot(hist):
    """The core technical calculation, factored out so both /analyze and /watchlist can reuse it."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


@app.get("/")
def health_check():
    return {"status": "ok", "message": "NSE Stock Analysis API v2 is running", "docs": "/docs"}


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    try:
        stock = yf.Ticker(ticker)
        hist = get_history(ticker, "2y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        technical, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = get_fundamentals(stock)

        return {
            "symbol": ticker,
            "company_name": ticker,
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years — "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag else None
            ),
            "technical": technical,
            "fundamentals": fundamentals,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}")
def chart_data(symbol: str, months: int = 6):
    """
    OHLC candlestick data + overlay indicator series for charting.
    Returns one row per trading day: date, OHLCV, RSI, MACD line/signal, MA50, MA200.
    `months` controls how much history to return in the chart (default 6, max 24).
    """
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    months = min(max(months, 1), 24)

    try:
        stock = yf.Ticker(ticker)
        hist = get_history(ticker, "2y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, _ = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)

        trading_days = months * 21  # approx trading days per month
        recent = hist.tail(trading_days)

        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]),
                "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]),
                "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]) if date in rsi.index else None,
                "macd_line": safe_round(macd_line.loc[date]) if date in macd_line.index else None,
                "macd_signal": safe_round(signal_line.loc[date]) if date in signal_line.index else None,
                "ma_50": safe_round(ma50.loc[date]) if date in ma50.index else None,
                "ma_200": safe_round(ma200.loc[date]) if date in ma200.index else None,
            })

        return {"symbol": ticker, "months": months, "data": rows}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist")
def watchlist(symbols: str):
    """
    Lightweight multi-stock snapshot for a watchlist view.
    Pass symbols comma-separated: /watchlist?symbols=RELIANCE,TCS,INFY
    Capped at 10 symbols per request to keep response time reasonable on free hosting.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        try:
            stock = yf.Ticker(ticker)
            hist = get_history(ticker, "2y")
            if hist.empty:
                results.append({"symbol": ticker, "error": "No data found"})
                continue

            technical, corporate_action_flag = get_technical_snapshot(hist)
            results.append({
                "symbol": ticker,
                "company_name": ticker,
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": corporate_action_flag,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    return {"count": len(results), "stocks": results}


# ══════════════════════════════════════════════════════════════════════════
#  NEW IN v2 — SEVEN ADDITIONAL FEATURES (all free: only yfinance + pandas + numpy)
# ══════════════════════════════════════════════════════════════════════════

# ── FEATURE 1: MARKET OVERVIEW (NIFTY 50, SENSEX, BANK NIFTY, NIFTY IT) ───

INDICES = {
    "^NSEI": "NIFTY 50",
    "^BSESN": "SENSEX",
    "^NSEBANK": "NIFTY BANK",
    "^CNXIT": "NIFTY IT",
}


@app.get("/market-overview")
def market_overview():
    """Latest level, daily change and 30-day sparkline for the main Indian indices."""
    indices = []
    for ticker, name in INDICES.items():
        try:
            hist = get_history(ticker, "3mo")
            if hist is None or hist.empty or len(hist) < 2:
                indices.append({"symbol": ticker, "name": name, "error": "No data"})
                continue
            close = hist["Close"]
            last, prev = close.iloc[-1], close.iloc[-2]
            month_base = close.iloc[-22] if len(close) >= 22 else close.iloc[0]
            indices.append({
                "symbol": ticker,
                "name": name,
                "level": safe_round(last),
                "change": safe_round(last - prev),
                "change_pct": safe_round((last / prev - 1) * 100),
                "change_1m_pct": safe_round((last / month_base - 1) * 100),
                "as_of": close.index[-1].strftime("%Y-%m-%d"),
                "sparkline": [safe_round(x) for x in close.tail(30)],
            })
        except Exception as e:
            indices.append({"symbol": ticker, "name": name, "error": str(e)})
    return {"timestamp": datetime.now().isoformat(), "indices": indices}


# ── FEATURE 2: SIGNAL SCORECARD (combines all indicators into one verdict) ─

@app.get("/signals/{symbol}")
def signal_scorecard(symbol: str):
    """
    Turns 7 indicators into a simple score from -7 (very bearish) to +7 (very bullish).
    Each indicator gives +1, 0 or -1 with a plain-English reason.
    Educational only — not investment advice.
    """
    ticker = to_ticker(symbol)
    try:
        hist = get_history(ticker, "2y")
        if hist is None or hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        close, high, low = hist["Close"], hist["High"], hist["Low"]
        price = close.iloc[-1]
        rsi = calculate_rsi(close).iloc[-1]
        roc = calculate_roc(close).iloc[-1]
        _, _, macd_hist = calculate_macd(close)
        macd_hist = macd_hist.iloc[-1]
        ma50_s, ma200_s = calculate_moving_averages(close)
        ma50, ma200 = ma50_s.iloc[-1], ma200_s.iloc[-1]
        adx = calculate_adx(high, low, close).iloc[-1]
        percent_b, _ = calculate_bollinger_bands(close)
        percent_b = percent_b.iloc[-1]

        checks = []

        def add(indicator, value, points, reason):
            checks.append({
                "indicator": indicator,
                "value": value,
                "signal": "bullish" if points > 0 else "bearish" if points < 0 else "neutral",
                "points": points,
                "reason": reason,
            })

        # 1. RSI
        if pd.isna(rsi):
            add("RSI (14)", None, 0, "Not enough data")
        elif rsi < 30:
            add("RSI (14)", safe_round(rsi), 1, "Oversold (below 30) — price may bounce")
        elif rsi > 70:
            add("RSI (14)", safe_round(rsi), -1, "Overbought (above 70) — price may cool off")
        else:
            add("RSI (14)", safe_round(rsi), 0, "In the normal 30–70 range")

        # 2. MACD histogram
        if pd.isna(macd_hist):
            add("MACD histogram", None, 0, "Not enough data")
        elif macd_hist > 0:
            add("MACD histogram", safe_round(macd_hist), 1, "MACD line is above its signal line — momentum is up")
        else:
            add("MACD histogram", safe_round(macd_hist), -1, "MACD line is below its signal line — momentum is down")

        # 3. Price vs 50-day average
        if pd.isna(ma50):
            add("Price vs MA 50", None, 0, "Not enough data")
        elif price > ma50:
            add("Price vs MA 50", safe_round(price - ma50), 1, "Price is above the 50-day average")
        else:
            add("Price vs MA 50", safe_round(price - ma50), -1, "Price is below the 50-day average")

        # 4. Golden / death alignment
        if pd.isna(ma50) or pd.isna(ma200):
            add("MA 50 vs MA 200", None, 0, "Not enough data for a 200-day average")
        elif ma50 > ma200:
            add("MA 50 vs MA 200", safe_round(ma50 - ma200), 1, "Golden alignment — long-term trend is up")
        else:
            add("MA 50 vs MA 200", safe_round(ma50 - ma200), -1, "Death alignment — long-term trend is down")

        # 5. ADX confirms whether the trend is strong enough to trust
        if pd.isna(adx) or pd.isna(ma50):
            add("ADX (14)", None, 0, "Not enough data")
        elif adx > 25:
            direction = 1 if price > ma50 else -1
            add("ADX (14)", safe_round(adx), direction,
                "Strong trend, and price is " + ("above" if direction > 0 else "below") + " its 50-day average")
        else:
            add("ADX (14)", safe_round(adx), 0, "Weak trend (ADX below 25) — trend signals are less reliable")

        # 6. Bollinger %B
        if pd.isna(percent_b):
            add("Bollinger %B", None, 0, "Not enough data")
        elif percent_b < 0.2:
            add("Bollinger %B", safe_round(percent_b, 3), 1, "Price is near the lower band — stretched down")
        elif percent_b > 0.8:
            add("Bollinger %B", safe_round(percent_b, 3), -1, "Price is near the upper band — stretched up")
        else:
            add("Bollinger %B", safe_round(percent_b, 3), 0, "Price is inside the middle of the bands")

        # 7. Rate of change
        if pd.isna(roc):
            add("ROC (12)", None, 0, "Not enough data")
        elif roc > 0:
            add("ROC (12)", safe_round(roc), 1, "Price is higher than 12 trading days ago")
        else:
            add("ROC (12)", safe_round(roc), -1, "Price is lower than 12 trading days ago")

        score = sum(c["points"] for c in checks)
        if score >= 4:
            verdict = "Strongly Bullish"
        elif score >= 2:
            verdict = "Bullish"
        elif score >= -1:
            verdict = "Neutral"
        elif score >= -3:
            verdict = "Bearish"
        else:
            verdict = "Strongly Bearish"

        return {
            "symbol": ticker,
            "score": score,
            "max_score": len(checks),
            "verdict": verdict,
            "bullish_count": sum(1 for c in checks if c["points"] > 0),
            "bearish_count": sum(1 for c in checks if c["points"] < 0),
            "neutral_count": sum(1 for c in checks if c["points"] == 0),
            "signals": checks,
            "disclaimer": "Educational technical-analysis score only. Not investment advice.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building signals for {ticker}: {str(e)}")


# ── FEATURE 3: RISK METRICS ────────────────────────────────────────────────

@app.get("/risk/{symbol}")
def risk_metrics(symbol: str, years: int = 2, risk_free_rate: float = 6.5):
    """Volatility, Sharpe, Sortino, max drawdown, VaR and beta (vs NIFTY 50) for one stock."""
    years = min(max(years, 1), 5)
    ticker = to_ticker(symbol)
    rf = risk_free_rate / 100
    try:
        hist = get_history(ticker, period_for_years(years))
        if hist is None or hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")
        close = hist["Close"].tail(years * 252)
        returns = close.pct_change().dropna()
        if len(returns) < 30:
            raise HTTPException(status_code=400, detail="Not enough history to calculate risk metrics.")

        stats = perf_stats(returns, rf)
        annual_return = (close.iloc[-1] / close.iloc[0]) ** (252 / len(returns)) - 1
        downside_vol = returns[returns < 0].std() * np.sqrt(252)
        sortino = (annual_return - rf) / downside_vol if downside_vol and downside_vol > 0 else None

        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        drawdown = close / close.cummax() - 1
        trough_date = drawdown.idxmin()
        peak_date = close.loc[:trough_date].idxmax()

        # Beta = how much the stock moves for a 1% move in NIFTY 50
        beta = None
        correlation = None
        try:
            bench = get_history("^NSEI", period_for_years(years))
            bench_ret = bench["Close"].pct_change().dropna()
            joined = pd.concat([returns, bench_ret], axis=1, join="inner").dropna()
            joined.columns = ["stock", "bench"]
            if len(joined) > 30 and joined["bench"].var() > 0:
                beta = safe_round(joined["stock"].cov(joined["bench"]) / joined["bench"].var())
                correlation = safe_round(joined["stock"].corr(joined["bench"]), 3)
        except Exception:
            pass

        dd_series = downsample(drawdown.to_frame("dd"))
        return {
            "symbol": ticker,
            "years": years,
            "risk_free_rate_pct": risk_free_rate,
            "annual_return_pct": safe_round(annual_return * 100),
            "annual_volatility_pct": stats.get("volatility_pct"),
            "sharpe_ratio": stats.get("sharpe_ratio"),
            "sortino_ratio": safe_round(sortino),
            "max_drawdown_pct": stats.get("max_drawdown_pct"),
            "max_drawdown_peak_date": peak_date.strftime("%Y-%m-%d"),
            "max_drawdown_trough_date": trough_date.strftime("%Y-%m-%d"),
            "var_95_daily_pct": safe_round(var_95 * 100),
            "cvar_95_daily_pct": safe_round(cvar_95 * 100),
            "beta_vs_nifty50": beta,
            "correlation_vs_nifty50": correlation,
            "best_day_pct": safe_round(returns.max() * 100),
            "worst_day_pct": safe_round(returns.min() * 100),
            "drawdown_series": [
                {"date": d.strftime("%Y-%m-%d"), "drawdown_pct": safe_round(v * 100)}
                for d, v in dd_series["dd"].items()
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating risk for {ticker}: {str(e)}")


# ── FEATURE 4: COMPARE STOCKS (rebased to 100) ────────────────────────────

@app.get("/compare")
def compare_stocks(symbols: str, months: int = 12):
    """
    Compare 2–5 stocks on one chart. Every price is rebased to 100 on day one,
    so you can see who actually performed better. Tip: add ^NSEI to compare with NIFTY 50.
    """
    months = min(max(months, 1), 60)
    names = parse_symbols(symbols, 2, 5)
    try:
        closes = load_closes([to_ticker(s) for s in names], months * 21)
        rebased = closes / closes.iloc[0] * 100
        returns = closes.pct_change().dropna()

        summary = []
        for col in closes.columns:
            summary.append({"symbol": col, **perf_stats(returns[col])})
        best = max(summary, key=lambda s: s.get("total_return_pct") or -1e9)
        worst = min(summary, key=lambda s: s.get("total_return_pct") or 1e9)

        series = []
        for date, row in downsample(rebased).iterrows():
            point = {"date": date.strftime("%Y-%m-%d")}
            for col in rebased.columns:
                point[col] = safe_round(row[col])
            series.append(point)

        return {
            "symbols": list(closes.columns),
            "months": months,
            "from": closes.index[0].strftime("%Y-%m-%d"),
            "to": closes.index[-1].strftime("%Y-%m-%d"),
            "best_performer": best["symbol"],
            "worst_performer": worst["symbol"],
            "summary": summary,
            "series": series,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing stocks: {str(e)}")


# ── FEATURE 5: CORRELATION MATRIX ──────────────────────────────────────────

@app.get("/correlation")
def correlation_matrix(symbols: str, months: int = 12):
    """How closely do these stocks move together? Low correlation = better diversification."""
    months = min(max(months, 3), 60)
    names = parse_symbols(symbols, 2, 8)
    try:
        closes = load_closes([to_ticker(s) for s in names], months * 21)
        returns = closes.pct_change().dropna()
        corr = returns.corr()
        cols = list(corr.columns)

        matrix = [[safe_round(corr.iloc[i, j], 3) for j in range(len(cols))] for i in range(len(cols))]
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append({"a": cols[i], "b": cols[j], "correlation": safe_round(corr.iloc[i, j], 3)})
        pairs.sort(key=lambda p: p["correlation"] if p["correlation"] is not None else -2, reverse=True)
        avg_corr = float(np.mean([p["correlation"] for p in pairs if p["correlation"] is not None]))

        if avg_corr > 0.7:
            note = "These stocks move very closely together — holding all of them gives little diversification."
        elif avg_corr > 0.4:
            note = "Moderate correlation — some diversification benefit."
        else:
            note = "Low correlation — a good diversification mix."

        return {
            "symbols": cols,
            "months": months,
            "matrix": matrix,
            "most_correlated": pairs[0],
            "least_correlated": pairs[-1],
            "average_correlation": safe_round(avg_corr, 3),
            "diversification_note": note,
            "pairs": pairs,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating correlation: {str(e)}")


# ── FEATURE 6: PORTFOLIO ANALYZER ──────────────────────────────────────────

@app.get("/portfolio")
def portfolio_analysis(symbols: str, weights: str = "", months: int = 24,
                       risk_free_rate: float = 6.5, investment: float = 100000):
    """
    Analyse a portfolio of 1–10 stocks.
    Example: /portfolio?symbols=TCS,INFY,HDFCBANK&weights=40,30,30&months=24
    Weights are in % and are rescaled to 100. If left out, the portfolio is equal-weighted.
    Assumes the weights are rebalanced every day (a standard simplification).
    """
    months = min(max(months, 3), 60)
    names = parse_symbols(symbols, 1, 10)
    rf = risk_free_rate / 100

    if weights.strip():
        try:
            w = np.array([float(x) for x in weights.split(",") if x.strip()])
        except ValueError:
            raise HTTPException(status_code=400, detail="Weights must be numbers, e.g. weights=40,30,30")
        if len(w) != len(names):
            raise HTTPException(status_code=400, detail=f"You gave {len(names)} different symbols but {len(w)} weights.")
        if (w < 0).any() or w.sum() <= 0:
            raise HTTPException(status_code=400, detail="Weights must be positive.")
    else:
        w = np.ones(len(names))
    w = w / w.sum()

    try:
        closes = load_closes([to_ticker(s) for s in names], months * 21)
        returns = closes.pct_change().dropna()
        port_ret = pd.Series(returns.values @ w, index=returns.index)
        stats = perf_stats(port_ret, rf)

        cov = returns.cov().values * 252
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(port_var)
        indiv_vol = np.sqrt(np.diag(cov))
        diversification_ratio = float((w * indiv_vol).sum() / port_vol) if port_vol > 0 else None
        risk_contrib = (w * (cov @ w)) / port_var if port_var > 0 else np.zeros(len(w))

        holdings = []
        for i, col in enumerate(returns.columns):
            holdings.append({
                "symbol": col,
                "weight_pct": safe_round(w[i] * 100),
                "total_return_pct": safe_round((closes[col].iloc[-1] / closes[col].iloc[0] - 1) * 100),
                "volatility_pct": safe_round(indiv_vol[i] * 100),
                "risk_contribution_pct": safe_round(risk_contrib[i] * 100),
            })

        growth = investment * (1 + port_ret).cumprod()
        curve = pd.DataFrame({"portfolio": growth})
        try:
            bench = get_history("^NSEI", period_for_years(months / 12))["Close"].pct_change().dropna()
            bench = bench.reindex(port_ret.index).fillna(0)
            curve["nifty50"] = investment * (1 + bench).cumprod()
        except Exception:
            pass

        equity = []
        for date, row in downsample(curve).iterrows():
            point = {"date": date.strftime("%Y-%m-%d")}
            for col in curve.columns:
                point[col] = safe_round(row[col], 0)
            equity.append(point)

        return {
            "months": months,
            "investment": investment,
            "final_value": safe_round(growth.iloc[-1], 0),
            "stats": stats,
            "diversification_ratio": safe_round(diversification_ratio),
            "best_day_pct": safe_round(port_ret.max() * 100),
            "worst_day_pct": safe_round(port_ret.min() * 100),
            "holdings": holdings,
            "equity_curve": equity,
            "note": "Assumes daily rebalancing to the chosen weights. Past performance does not predict future returns.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analysing portfolio: {str(e)}")


# ── FEATURE 7: STRATEGY BACKTESTER ─────────────────────────────────────────

@app.get("/backtest/{symbol}")
def backtest(symbol: str, strategy: str = "ma_crossover", years: int = 3,
             fast: int = 50, slow: int = 200, lookback: int = 126,
             cost_pct: float = 0.1, risk_free_rate: float = 6.5):
    """
    Test a simple strategy on past prices and compare with buy-and-hold.
      strategy = ma_crossover  -> hold the stock while MA(fast) > MA(slow), otherwise cash
      strategy = momentum      -> hold the stock while the return over `lookback` days is positive
      strategy = rsi_reversion -> buy when RSI < 30, sell when RSI > 50
    The signal of today is traded at tomorrow's price (no look-ahead cheating).
    cost_pct is the cost per trade in % (brokerage + slippage).
    """
    strategies = ("ma_crossover", "momentum", "rsi_reversion")
    if strategy not in strategies:
        raise HTTPException(status_code=400, detail=f"strategy must be one of: {', '.join(strategies)}")
    years = min(max(years, 1), 8)
    if strategy == "ma_crossover" and not (2 <= fast < slow <= 400):
        raise HTTPException(status_code=400, detail="For ma_crossover, fast must be smaller than slow (max 400).")
    lookback = min(max(lookback, 5), 400)
    rf = risk_free_rate / 100
    ticker = to_ticker(symbol)

    try:
        # fetch one extra year so slow indicators are "warmed up" before the test window starts
        hist = get_history(ticker, period_for_years(years + 1))
        if hist is None or hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        close = hist["Close"]
        daily_ret = close.pct_change()

        if strategy == "ma_crossover":
            signal = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(int)
        elif strategy == "momentum":
            signal = (close.pct_change(lookback) > 0).astype(int)
        else:
            rsi = calculate_rsi(close)
            state, values = 0, []
            for v in rsi.values:
                if state == 0 and v < 30:
                    state = 1
                elif state == 1 and v > 50:
                    state = 0
                values.append(state)
            signal = pd.Series(values, index=close.index)

        position = signal.shift(1).fillna(0)                      # trade next day
        costs = position.diff().abs().fillna(0) * (cost_pct / 100)
        strat_ret = position * daily_ret - costs

        n_eval = years * 252
        pos_e = position.tail(n_eval)
        strat_e = strat_ret.tail(n_eval).dropna()
        bh_e = daily_ret.tail(n_eval).dropna()
        if len(strat_e) < 60:
            raise HTTPException(status_code=400, detail="Not enough history to run this backtest.")

        # individual trades
        seg_id = (pos_e != pos_e.shift()).cumsum()
        trade_returns = []
        for _, seg in strat_ret.tail(n_eval).groupby(seg_id):
            if pos_e.loc[seg.index[0]] == 1:
                trade_returns.append((1 + seg).prod() - 1)
        wins = [t for t in trade_returns if t > 0]

        curve = pd.DataFrame({
            "strategy": 100 * (1 + strat_e).cumprod(),
            "buy_and_hold": 100 * (1 + bh_e).cumprod(),
        })
        equity = []
        for date, row in downsample(curve).iterrows():
            equity.append({
                "date": date.strftime("%Y-%m-%d"),
                "strategy": safe_round(row["strategy"]),
                "buy_and_hold": safe_round(row["buy_and_hold"]),
            })

        strat_stats = perf_stats(strat_e, rf)
        bh_stats = perf_stats(bh_e, rf)
        beat = (strat_stats.get("total_return_pct") or 0) > (bh_stats.get("total_return_pct") or 0)

        return {
            "symbol": ticker,
            "strategy": strategy,
            "parameters": {"years": years, "fast": fast, "slow": slow, "lookback": lookback, "cost_pct": cost_pct},
            "strategy_stats": strat_stats,
            "buy_and_hold_stats": bh_stats,
            "number_of_trades": len(trade_returns),
            "win_rate_pct": safe_round(len(wins) / len(trade_returns) * 100) if trade_returns else None,
            "avg_trade_return_pct": safe_round(float(np.mean(trade_returns)) * 100) if trade_returns else None,
            "time_in_market_pct": safe_round(pos_e.mean() * 100),
            "current_position": "invested" if signal.iloc[-1] == 1 else "cash",
            "beat_buy_and_hold": bool(beat),
            "equity_curve": equity,
            "note": "Backtests use past data and ignore taxes. They do not guarantee future results.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running backtest for {ticker}: {str(e)}")