import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone


# ============================================================
# QUANT PRO V4
# Professional US Stock Quant Dashboard
# ============================================================

st.set_page_config(
    page_title="QUANT PRO V4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# UI
# ============================================================

st.markdown("""
<style>
:root{
    --bg:#080d12;
    --panel:#0f1720;
    --panel2:#131d28;
    --line:#223244;
    --text:#e9f0f7;
    --muted:#89a0b6;
    --green:#31d17c;
    --red:#ff6673;
    --yellow:#f4c655;
    --blue:#5ba8ff;
    --cyan:#52d3e8;
}

html,body,[class*="css"]{
    font-size:12px;
}

.stApp{
    background:linear-gradient(180deg,#070b10 0%,#0b1118 100%);
    color:var(--text);
}

.block-container{
    max-width:1200px!important;
    padding:.55rem .72rem 1.5rem!important;
}

h1{font-size:1.22rem!important;margin:0!important;}
h2{font-size:.92rem!important;margin:.65rem 0 .35rem!important;}
h3{font-size:.82rem!important;}
p,label,.stCaption{font-size:.72rem!important;}

[data-testid="stMetric"]{
    background:linear-gradient(180deg,#111a24,#0e161e);
    border:1px solid var(--line);
    border-radius:10px;
    padding:8px 9px;
    min-height:66px;
}

[data-testid="stMetric"] label{
    font-size:.61rem!important;
    color:var(--muted)!important;
}

[data-testid="stMetricValue"]{
    font-size:1rem!important;
    line-height:1.05!important;
}

[data-testid="stMetricDelta"]{
    font-size:.6rem!important;
}

.stButton>button{
    width:100%;
    min-height:34px;
    border-radius:8px;
    border:1px solid var(--line);
    background:#101923;
    color:var(--text);
    font-size:.70rem;
}

div[data-baseweb="select"]>div,
.stTextInput input,
.stNumberInput input{
    background:#0e161f!important;
    border-color:var(--line)!important;
    min-height:35px!important;
    font-size:.74rem!important;
}

[data-testid="stTabs"] button{
    font-size:.72rem!important;
    padding:.45rem .55rem!important;
}

.hero{
    display:flex;
    justify-content:space-between;
    gap:8px;
    align-items:flex-start;
    margin:0 0 6px 0;
}

.hero-title{
    font-size:1.12rem;
    font-weight:850;
    letter-spacing:.25px;
}

.hero-sub{
    font-size:.61rem;
    color:var(--muted);
    margin-top:1px;
}

.pill{
    border:1px solid var(--line);
    border-radius:999px;
    padding:4px 7px;
    color:var(--muted);
    background:#0e161f;
    font-size:.58rem;
    white-space:nowrap;
}

.signal{
    border:1px solid var(--line);
    border-radius:11px;
    padding:10px 11px;
    margin:5px 0 7px;
    background:linear-gradient(180deg,#111a24,#0e161f);
}

.signal.long{border-left:4px solid var(--green);}
.signal.short{border-left:4px solid var(--red);}
.signal.wait{border-left:4px solid var(--yellow);}
.signal.none{border-left:4px solid #718096;}

.signal-title{
    font-size:.96rem;
    font-weight:850;
}

.signal-sub{
    font-size:.61rem;
    color:var(--muted);
    margin-top:2px;
}

.section{
    font-size:.66rem;
    color:#b7c6d5;
    font-weight:800;
    letter-spacing:.18px;
    margin:8px 0 4px;
}

.grid4{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:6px;
}

.card{
    background:#0f1720;
    border:1px solid var(--line);
    border-radius:9px;
    padding:7px 8px;
}

.card .k{
    font-size:.56rem;
    color:var(--muted);
    margin-bottom:2px;
}

.card .v{
    font-size:.82rem;
    font-weight:780;
}

.card .s{
    font-size:.53rem;
    color:var(--muted);
    margin-top:1px;
}

.g{color:var(--green);}
.r{color:var(--red);}
.y{color:var(--yellow);}
.b{color:var(--blue);}

.reason{
    background:#0e161f;
    border:1px solid var(--line);
    border-radius:8px;
    padding:6px 8px;
    margin-bottom:4px;
    font-size:.63rem;
}

.rank-row{
    display:grid;
    grid-template-columns:.6fr .65fr .7fr .7fr .7fr;
    gap:4px;
    align-items:center;
    background:#0f1720;
    border:1px solid var(--line);
    border-radius:8px;
    padding:6px 7px;
    margin-bottom:4px;
    font-size:.62rem;
}

.rank-head{
    color:var(--muted);
    background:transparent;
    border:none;
    font-weight:700;
}

.foot{
    font-size:.54rem;
    color:var(--muted);
    text-align:center;
    margin-top:8px;
}

@media(max-width:700px){
    .grid4{
        grid-template-columns:repeat(2,1fr);
    }

    .block-container{
        padding-left:.55rem!important;
        padding-right:.55rem!important;
    }

    [data-testid="stMetric"]{
        min-height:62px;
    }

    [data-testid="stMetricValue"]{
        font-size:.92rem!important;
    }

    .rank-row{
        grid-template-columns:.65fr .65fr .8fr .65fr .7fr;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def fmt(x, d=2, prefix=""):
    x = safe_float(x)
    return "-" if not np.isfinite(x) else f"{prefix}{x:.{d}f}"


def trend_cn(x):
    return {
        "BULL": "多",
        "BEAR": "空",
        "NEUTRAL": "中性"
    }.get(x, x)


def leg_cn(x):
    return {
        "FIRST_PUSH": "第一推动",
        "SECOND_PUSH": "第二推动",
        "THIRD_PUSH": "第三推动"
    }.get(x, x)


# ============================================================
# INDICATORS
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    pc = df["Close"].shift(1)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/n, adjust=False).mean()


def adx(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where((up > dn) & (up > 0), up, 0.0),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where((dn > up) & (dn > 0), dn, 0.0),
        index=df.index
    )

    a = atr(df, n)

    plus_di = (
        100 *
        plus_dm.ewm(alpha=1/n, adjust=False).mean() /
        a.replace(0, np.nan)
    )

    minus_di = (
        100 *
        minus_dm.ewm(alpha=1/n, adjust=False).mean() /
        a.replace(0, np.nan)
    )

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(0, np.nan)
    )

    adx_value = dx.ewm(alpha=1/n, adjust=False).mean()

    return adx_value, plus_di, minus_di


def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = pd.Series(df.index.date, index=df.index)
    pv = tp * df["Volume"]

    return (
        pv.groupby(day).cumsum() /
        df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    )


def add_indicators(df):
    x = df.copy()

    x["EMA20"] = ema(x["Close"], 20)
    x["EMA60"] = ema(x["Close"], 60)

    x["RSI"] = rsi(x["Close"])
    x["ATR"] = atr(x)
    x["VWAP"] = vwap(x)

    macd = ema(x["Close"], 12) - ema(x["Close"], 26)
    x["MACD_H"] = macd - ema(macd, 9)

    x["ADX"], x["PLUS_DI"], x["MINUS_DI"] = adx(x)

    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["RVOL"] = x["Volume"] / x["VOL20"].replace(0, np.nan)

    x["OBV"] = (
        np.sign(x["Close"].diff()).fillna(0) * x["Volume"]
    ).cumsum()

    x["HH20"] = x["High"].shift(1).rolling(20).max()
    x["LL20"] = x["Low"].shift(1).rolling(20).min()

    x["EMA20_SLOPE"] = (
        x["EMA20"].diff(5) /
        x["ATR"].replace(0, np.nan)
    )

    x["BB_MID"] = x["Close"].rolling(20).mean()
    bb_std = x["Close"].rolling(20).std()

    x["BB_UP"] = x["BB_MID"] + 2 * bb_std
    x["BB_LOW"] = x["BB_MID"] - 2 * bb_std

    x["BB_WIDTH"] = (
        (x["BB_UP"] - x["BB_LOW"]) /
        x["BB_MID"].replace(0, np.nan)
    )

    x["ROC10"] = x["Close"].pct_change(10) * 100

    x["RET"] = x["Close"].pct_change()

    x["VOLATILITY"] = (
        x["RET"].rolling(20).std() *
        np.sqrt(78) *
        100
    )

    return x


# ============================================================
# RESAMPLE
# ============================================================

def resample_ohlcv(df, rule):
    return df.resample(rule).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum")
    ).dropna()


def trend_state(df):
    if df is None or len(df) < 65:
        return "NEUTRAL"

    r = df.iloc[-1]

    if r["Close"] > r["EMA20"] > r["EMA60"]:
        return "BULL"

    if r["Close"] < r["EMA20"] < r["EMA60"]:
        return "BEAR"

    return "NEUTRAL"


def detect_leg(df, side):
    recent = df.tail(30)

    if side == "LONG":
        pushes = (
            recent["High"] >= recent["High"].rolling(6).max()
        ).sum()
    else:
        pushes = (
            recent["Low"] <= recent["Low"].rolling(6).min()
        ).sum()

    if pushes >= 9:
        return "THIRD_PUSH"

    if pushes >= 6:
        return "SECOND_PUSH"

    return "FIRST_PUSH"


# ============================================================
# ALPACA
# ============================================================

def alpaca_ready():
    try:
        return (
            bool(st.secrets.get("ALPACA_API_KEY"))
            and
            bool(st.secrets.get("ALPACA_SECRET_KEY"))
        )
    except Exception:
        return False


def alpaca_headers():
    return {
        "APCA-API-KEY-ID": st.secrets["ALPACA_API_KEY"],
        "APCA-API-SECRET-KEY": st.secrets["ALPACA_SECRET_KEY"],
    }


@st.cache_data(ttl=25, show_spinner=False)
def load_alpaca(symbol, days=30, timeframe="5Min"):
    start = (
        datetime.now(timezone.utc) -
        timedelta(days=days)
    ).isoformat()

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"

    params = {
        "timeframe": timeframe,
        "start": start,
        "limit": 10000,
        "adjustment": "raw",
        "feed": "iex",
        "sort": "asc",
    }

    r = requests.get(
        url,
        headers=alpaca_headers(),
        params=params,
        timeout=20
    )

    r.raise_for_status()

    bars = r.json().get("bars", [])

    if not bars:
        return None

    x = pd.DataFrame(bars)

    x["t"] = pd.to_datetime(x["t"], utc=True)

    x = x.set_index("t").rename(columns={
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume"
    })

    return x[
        ["Open", "High", "Low", "Close", "Volume"]
    ].astype(float).dropna()


# ============================================================
# YAHOO
# ============================================================

@st.cache_data(ttl=45, show_spinner=False)
def load_yahoo(symbol, period="30d", interval="5m"):
    x = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=False
    )

    if x is None or x.empty:
        return None

    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)

    need = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in x.columns for c in need):
        return None

    return x[need].dropna()


def load_symbol(symbol, provider="AUTO"):
    if provider in ("AUTO", "ALPACA") and alpaca_ready():

        try:
            x = load_alpaca(symbol)

            if x is not None and len(x) > 150:
                return x, "ALPACA"

        except Exception:
            if provider == "ALPACA":
                raise

    return load_yahoo(symbol), "YAHOO"


@st.cache_data(ttl=120, show_spinner=False)
def load_daily_yahoo(symbol):
    x = yf.download(
        symbol,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if x is None or x.empty:
        return None

    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)

    need = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in x.columns for c in need):
        return None

    return x[need].dropna()


# ============================================================
# MARKET REGIME
# ============================================================

def market_regime(provider="AUTO"):
    score = 0
    details = []

    for sym in ["SPY", "QQQ", "IWM"]:

        d, _ = load_symbol(sym, provider)

        if d is None or len(d) < 70:
            details.append(f"{sym}无数据")
            continue

        f = add_indicators(d)
        r = f.iloc[-1]

        if r["Close"] > r["EMA20"] and r["Close"] > r["VWAP"]:
            score += 1
            details.append(f"{sym}多")

        elif r["Close"] < r["EMA20"] and r["Close"] < r["VWAP"]:
            score -= 1
            details.append(f"{sym}空")

        else:
            details.append(f"{sym}中")

    vix = load_yahoo("^VIX", period="5d", interval="15m")

    vix_last = np.nan

    if vix is not None and len(vix):

        vix_last = safe_float(vix["Close"].iloc[-1])

        if vix_last >= 30:
            score -= 2

        elif vix_last >= 25:
            score -= 1

        elif vix_last <= 17:
            score += 1

    if score >= 2:
        bias = "BULL"

    elif score <= -2:
        bias = "BEAR"

    else:
        bias = "NEUTRAL"

    return bias, " / ".join(details), vix_last, score


# ============================================================
# SIGNAL ENGINE V4
# ============================================================

def evaluate(df, daily_df, market_bias):

    if df is None or len(df) < 200:
        return None

    f5 = add_indicators(df)

    f15 = add_indicators(
        resample_ohlcv(df, "15min")
    )

    f60 = add_indicators(
        resample_ohlcv(df, "60min")
    )

    if min(len(f15), len(f60)) < 60:
        return None

    daily_trend = "NEUTRAL"

    if daily_df is not None and len(daily_df) >= 65:
        daily_trend = trend_state(
            add_indicators(daily_df)
        )

    r = f5.iloc[-1]
    p = f5.iloc[-2]

    t5 = trend_state(f5)
    t15 = trend_state(f15)
    t60 = trend_state(f60)

    L = 0
    S = 0

    lr = []
    sr = []

    # DAILY
    if daily_trend == "BULL":
        L += 10
        lr.append("日线趋势向上 +10")

    elif daily_trend == "BEAR":
        S += 10
        sr.append("日线趋势向下 +10")

    # 60m
    if t60 == "BULL":
        L += 18
        lr.append("60m主趋势向上 +18")

    elif t60 == "BEAR":
        S += 18
        sr.append("60m主趋势向下 +18")

    # 15m
    if t15 == "BULL":
        L += 12
        lr.append("15m趋势确认向上 +12")

    elif t15 == "BEAR":
        S += 12
        sr.append("15m趋势确认向下 +12")

    # EMA
    if r["Close"] > r["EMA20"] > r["EMA60"]:
        L += 10
        lr.append("5m EMA多头排列 +10")

    elif r["Close"] < r["EMA20"] < r["EMA60"]:
        S += 10
        sr.append("5m EMA空头排列 +10")

    # EMA SLOPE
    if pd.notna(r["EMA20_SLOPE"]):

        if r["EMA20_SLOPE"] > 0.25:
            L += 4
            lr.append("EMA20斜率向上 +4")

        elif r["EMA20_SLOPE"] < -0.25:
            S += 4
            sr.append("EMA20斜率向下 +4")

    # VWAP
    if r["Close"] > r["VWAP"]:
        L += 8
        lr.append("价格在VWAP上方 +8")

    elif r["Close"] < r["VWAP"]:
        S += 8
        sr.append("价格在VWAP下方 +8")

    # MACD
    if r["MACD_H"] > 0 and r["MACD_H"] > p["MACD_H"]:
        L += 8
        lr.append("MACD多头动能增强 +8")

    elif r["MACD_H"] < 0 and r["MACD_H"] < p["MACD_H"]:
        S += 8
        sr.append("MACD空头动能增强 +8")

    # RSI
    if 52 <= r["RSI"] <= 72:
        L += 7
        lr.append("RSI有效多头区 +7")

    elif 28 <= r["RSI"] <= 48:
        S += 7
        sr.append("RSI有效空头区 +7")

    # RVOL
    if pd.notna(r["RVOL"]) and r["RVOL"] >= 1.20:

        if r["Close"] > r["Open"]:
            L += 8
            lr.append(f"RVOL {r['RVOL']:.2f} 放量上涨 +8")

        elif r["Close"] < r["Open"]:
            S += 8
            sr.append(f"RVOL {r['RVOL']:.2f} 放量下跌 +8")

    # OBV
    if f5["OBV"].iloc[-1] > f5["OBV"].iloc[-6]:
        L += 4
        lr.append("OBV资金流上行 +4")

    elif f5["OBV"].iloc[-1] < f5["OBV"].iloc[-6]:
        S += 4
        sr.append("OBV资金流下行 +4")

    # BREAKOUT
    bull_break = (
        pd.notna(r["HH20"]) and
        r["Close"] > r["HH20"]
    )

    bear_break = (
        pd.notna(r["LL20"]) and
        r["Close"] < r["LL20"]
    )

    if bull_break:
        L += 10
        lr.append("突破20根K线前高 +10")

    if bear_break:
        S += 10
        sr.append("跌破20根K线前低 +10")

    # ROC
    if pd.notna(r["ROC10"]):

        if r["ROC10"] >= 1.0:
            L += 6
            lr.append(f"10周期动量 +{r['ROC10']:.2f}% +6")

        elif r["ROC10"] <= -1.0:
            S += 6
            sr.append(f"10周期动量 {r['ROC10']:.2f}% +6")

    # BOLLINGER
    if pd.notna(r["BB_UP"]) and r["Close"] > r["BB_UP"]:
        L += 5
        lr.append("价格突破布林上轨 +5")

    elif pd.notna(r["BB_LOW"]) and r["Close"] < r["BB_LOW"]:
        S += 5
        sr.append("价格跌破布林下轨 +5")

    # ADX
    if pd.notna(r["ADX"]) and r["ADX"] >= 25:

        if r["PLUS_DI"] > r["MINUS_DI"]:
            L += 5
            lr.append("ADX强趋势且+DI占优 +5")

        else:
            S += 5
            sr.append("ADX强趋势且-DI占优 +5")

    # MARKET
    if market_bias == "BULL":
        L += 8
        S -= 5
        lr.append("大盘环境偏多 +8")

    elif market_bias == "BEAR":
        S += 8
        L -= 5
        sr.append("大盘环境偏空 +8")

    # FOUR-TIMEFRAME ALIGNMENT
    if (
        daily_trend == "BULL"
        and t60 == "BULL"
        and t15 == "BULL"
        and t5 == "BULL"
    ):
        L += 8
        lr.append("日线/60m/15m/5m 四周期共振 +8")

    elif (
        daily_trend == "BEAR"
        and t60 == "BEAR"
        and t15 == "BEAR"
        and t5 == "BEAR"
    ):
        S += 8
        sr.append("日线/60m/15m/5m 四周期共振 +8")

    # FAKE BREAKOUT
    if bull_break and (
        pd.isna(r["RVOL"]) or r["RVOL"] < 1.05
    ):
        L -= 6
        lr.append("突破但成交量不足 -6")

    if bear_break and (
        pd.isna(r["RVOL"]) or r["RVOL"] < 1.05
    ):
        S -= 6
        sr.append("跌破但成交量不足 -6")

    # TIMEFRAME CONFLICT
    if (
        (t60 == "BULL" and t15 == "BEAR")
        or
        (t60 == "BEAR" and t15 == "BULL")
    ):
        L -= 10
        S -= 10

        lr.append("60m与15m趋势冲突 -10")
        sr.append("60m与15m趋势冲突 -10")

    long_leg = detect_leg(f5, "LONG")
    short_leg = detect_leg(f5, "SHORT")

    # OVEREXTENDED
    if long_leg == "THIRD_PUSH" and r["RSI"] > 72:
        L -= 8
        lr.append("第三推动 + RSI过热 -8")

    if short_leg == "THIRD_PUSH" and r["RSI"] < 28:
        S -= 8
        sr.append("第三推动 + RSI过冷 -8")

    # VOLATILITY
    vol_value = safe_float(r["VOLATILITY"])

    if np.isfinite(vol_value) and vol_value > 8:
        L -= 4
        S -= 4

        lr.append("短周期波动率过高 -4")
        sr.append("短周期波动率过高 -4")

    L = max(0, int(L))
    S = max(0, int(S))

    side = "LONG" if L >= S else "SHORT"

    score = max(L, S)
    gap = abs(L - S)

    # SIGNAL LEVEL
    if score >= 90 and gap >= 20:
        status = "CONFIRMED"

    elif score >= 82 and gap >= 15:
        status = "WAIT_CONFIRM"

    elif score >= 70 and gap >= 10:
        status = "WATCH"

    else:
        status = "NO_TRADE"

    if gap < 10:
        status = "NO_TRADE"

    entry = safe_float(r["Close"])
    a = safe_float(r["ATR"])

    stop = np.nan
    t1 = np.nan
    t2 = np.nan
    t3 = np.nan

    if np.isfinite(a) and a > 0:

        if side == "LONG":

            recent_low = safe_float(
                f5["Low"].tail(8).min()
            )

            stop = min(
                recent_low,
                entry - 1.35 * a
            )

            risk = entry - stop

            t1 = entry + risk
            t2 = entry + 2 * risk
            t3 = entry + 3 * risk

        else:

            recent_high = safe_float(
                f5["High"].tail(8).max()
            )

            stop = max(
                recent_high,
                entry + 1.35 * a
            )

            risk = stop - entry

            t1 = entry - risk
            t2 = entry - 2 * risk
            t3 = entry - 3 * risk

    return {
        "side": side,
        "status": status,
        "score": score,
        "gap": gap,

        "long_score": L,
        "short_score": S,

        "entry": entry,
        "stop": stop,

        "t1": t1,
        "t2": t2,
        "t3": t3,

        "trend5": t5,
        "trend15": t15,
        "trend60": t60,
        "daily": daily_trend,

        "rsi": safe_float(r["RSI"]),
        "adx": safe_float(r["ADX"]),
        "rvol": safe_float(r["RVOL"]),
        "vwap": safe_float(r["VWAP"]),
        "atr": a,

        "macd_h": safe_float(r["MACD_H"]),
        "roc10": safe_float(r["ROC10"]),
        "volatility": safe_float(r["VOLATILITY"]),
        "bb_width": safe_float(r["BB_WIDTH"]),

        "leg": long_leg if side == "LONG" else short_leg,

        "reasons": (
            lr if side == "LONG" else sr
        )[:12],

        "f5": f5.tail(120)
    }


# ============================================================
# LABEL
# ============================================================

def label(sig):

    if sig["status"] == "CONFIRMED":

        if sig["side"] == "LONG":
            return "🟢 强做多确认", "long"

        return "🔴 强做空确认", "short"

    if sig["status"] == "WAIT_CONFIRM":

        if sig["side"] == "LONG":
            return "🟡 多头等待确认", "wait"

        return "🟡 空头等待确认", "wait"

    if sig["status"] == "WATCH":

        if sig["side"] == "LONG":
            return "👀 多头观察", "wait"

        return "👀 空头观察", "wait"

    return "⚪ 禁止交易 / 信号冲突", "none"


# ============================================================
# POSITION SIZE
# ============================================================

def position_size(account, risk_pct, max_pos_pct, sig):

    if not np.isfinite(sig["stop"]):
        return 0, 0, 0

    per_share = abs(
        sig["entry"] - sig["stop"]
    )

    if per_share <= 0:
        return 0, 0, 0

    by_risk = int(
        (account * risk_pct) //
        per_share
    )

    by_notional = int(
        (account * max_pos_pct) //
        sig["entry"]
    )

    shares = max(
        0,
        min(
            by_risk,
            by_notional
        )
    )

    risk_value = shares * per_share
    notional = shares * sig["entry"]

    return shares, risk_value, notional


# ============================================================
# CHART
# ============================================================

def make_chart(sig):

    f = sig["f5"]

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=f.index,
            open=f["Open"],
            high=f["High"],
            low=f["Low"],
            close=f["Close"],
            name="Price",
            increasing_line_color="#31d17c",
            decreasing_line_color="#ff6673"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=f.index,
            y=f["EMA20"],
            name="EMA20",
            line=dict(width=1.1)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=f.index,
            y=f["EMA60"],
            name="EMA60",
            line=dict(width=1.1)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=f.index,
            y=f["VWAP"],
            name="VWAP",
            line=dict(
                width=1,
                dash="dot"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=f.index,
            y=f["BB_UP"],
            name="BB Upper",
            line=dict(
                width=.7,
                dash="dot"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=f.index,
            y=f["BB_LOW"],
            name="BB Lower",
            line=dict(
                width=.7,
                dash="dot"
            )
        )
    )

    fig.update_layout(
        height=330,

        margin=dict(
            l=0,
            r=0,
            t=8,
            b=0
        ),

        paper_bgcolor="#0b1118",
        plot_bgcolor="#0b1118",

        font=dict(
            size=9,
            color="#9fb0c2"
        ),

        xaxis_rangeslider_visible=False,

        legend=dict(
            orientation="h",
            y=1.02,
            x=0,
            font=dict(size=8)
        ),

        xaxis=dict(
            gridcolor="#182431"
        ),

        yaxis=dict(
            gridcolor="#182431"
        )
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div>
<div class="hero-title">
📈 QUANT PRO V4
</div>

<div class="hero-sub">
Multi-Timeframe Signal Engine · Scanner · Risk Control
</div>
</div>

<div class="pill">
5m LIVE ENGINE
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# CONTROLS
# ============================================================

c1, c2, c3, c4 = st.columns(
    [1.0, 1.1, .85, .5]
)

with c1:

    symbol = st.selectbox(
        "标的",
        [
            "NVDA",
            "TSLA",
            "AMZN",
            "GOOGL",
            "ORCL",
            "META",
            "AMD",
            "AVGO",
            "MSFT",
            "TSM",
            "PLTR",
            "NFLX",
            "AAPL"
        ],
        index=4
    )

with c2:

    custom = st.text_input(
        "自定义代码",
        placeholder="例如 MU / COIN"
    )

    if custom.strip():
        symbol = custom.strip().upper()

with c3:

    provider_choice = st.selectbox(
        "数据源",
        [
            "AUTO",
            "YAHOO",
            "ALPACA"
        ]
    )

with c4:

    st.write("")
    st.write("")

    if st.button("刷新"):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# RISK CONTROLS
# ============================================================

with st.expander(
    "⚙️ 风险参数",
    expanded=False
):

    a, b, c, d = st.columns(4)

    with a:

        account = st.number_input(
            "账户资金 $",
            min_value=1000.0,
            value=20000.0,
            step=1000.0
        )

    with b:

        risk_pct = (
            st.slider(
                "单笔风险 %",
                0.1,
                1.5,
                0.5,
                0.1
            ) / 100
        )

    with c:

        max_pos_pct = (
            st.slider(
                "最大仓位 %",
                5,
                50,
                25,
                5
            ) / 100
        )

    with d:

        day_loss_pct = (
            st.slider(
                "日内熔断 %",
                0.5,
                5.0,
                2.0,
                0.5
            ) / 100
        )


if provider_choice == "ALPACA" and not alpaca_ready():

    st.warning(
        "ALPACA 尚未配置。"
        "当前将自动回退 Yahoo 数据。"
    )


# ============================================================
# MARKET
# ============================================================

with st.spinner("读取市场环境..."):

    mbias, mdetail, vix, mkt_score = market_regime(
        provider_choice
    )


tabs = st.tabs(
    [
        "仪表盘",
        "扫描器",
        "市场",
        "风险"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tabs[0]:

    with st.spinner(
        f"分析 {symbol}..."
    ):

        df, used_provider = load_symbol(
            symbol,
            provider_choice
        )

        daily = load_daily_yahoo(
            symbol
        )

    if df is None:

        st.error(
            "无法获取行情，请稍后刷新。"
        )

    else:

        sig = evaluate(
            df,
            daily,
            mbias
        )

        if sig is None:

            st.warning(
                "数据不足，暂时无法生成信号。"
            )

        else:

            text, cls = label(sig)

            key = (
                f"{symbol}_"
                f"{sig['side']}_"
                f"{sig['status']}"
            )

            if "first_seen" not in st.session_state:
                st.session_state.first_seen = {}

            if key not in st.session_state.first_seen:

                st.session_state.first_seen[key] = (
                    datetime.now().strftime(
                        "%H:%M:%S"
                    )
                )

            first_seen = (
                st.session_state.first_seen[key]
            )

            st.markdown(
                f"""
                <div class="signal {cls}">
                <div class="signal-title">
                {text} · {symbol}
                </div>

                <div class="signal-sub">
                评分 {sig['score']} ·
                多 {sig['long_score']} /
                空 {sig['short_score']} ·
                Gap {sig['gap']} ·
                首次出现 {first_seen} ·
                数据源 {used_provider}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # CORE
            st.markdown(
                '<div class="section">核心概览</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="grid4">

                <div class="card">
                <div class="k">当前价格</div>
                <div class="v">${fmt(sig['entry'])}</div>
                <div class="s">Last 5m</div>
                </div>

                <div class="card">
                <div class="k">信号评分</div>
                <div class="v">{sig['score']}</div>
                <div class="s">{sig['status']}</div>
                </div>

                <div class="card">
                <div class="k">多 / 空</div>
                <div class="v">
                <span class="g">{sig['long_score']}</span>
                /
                <span class="r">{sig['short_score']}</span>
                </div>
                <div class="s">Gap {sig['gap']}</div>
                </div>

                <div class="card">
                <div class="k">推动阶段</div>
                <div class="v">{leg_cn(sig['leg'])}</div>
                <div class="s">Structure</div>
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # TIMEFRAMES
            st.markdown(
                '<div class="section">多周期共振</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="grid4">

                <div class="card">
                <div class="k">5分钟</div>
                <div class="v">{trend_cn(sig['trend5'])}</div>
                <div class="s">Execution</div>
                </div>

                <div class="card">
                <div class="k">15分钟</div>
                <div class="v">{trend_cn(sig['trend15'])}</div>
                <div class="s">Confirm</div>
                </div>

                <div class="card">
                <div class="k">60分钟</div>
                <div class="v">{trend_cn(sig['trend60'])}</div>
                <div class="s">Primary</div>
                </div>

                <div class="card">
                <div class="k">日线</div>
                <div class="v">{trend_cn(sig['daily'])}</div>
                <div class="s">Macro</div>
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # TRADE STRUCTURE
            st.markdown(
                '<div class="section">交易结构</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="grid4">

                <div class="card">
                <div class="k">入场参考</div>
                <div class="v">${fmt(sig['entry'])}</div>
                </div>

                <div class="card">
                <div class="k">结构止损</div>
                <div class="v r">${fmt(sig['stop'])}</div>
                </div>

                <div class="card">
                <div class="k">1R</div>
                <div class="v">${fmt(sig['t1'])}</div>
                </div>

                <div class="card">
                <div class="k">2R</div>
                <div class="v">${fmt(sig['t2'])}</div>
                </div>

                <div class="card">
                <div class="k">3R</div>
                <div class="v">${fmt(sig['t3'])}</div>
                </div>

                <div class="card">
                <div class="k">VWAP</div>
                <div class="v">${fmt(sig['vwap'])}</div>
                </div>

                <div class="card">
                <div class="k">RVOL</div>
                <div class="v">{fmt(sig['rvol'])}</div>
                </div>

                <div class="card">
                <div class="k">ATR</div>
                <div class="v">{fmt(sig['atr'])}</div>
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            shares, est_risk, notional = position_size(
                account,
                risk_pct,
                max_pos_pct,
                sig
            )

            st.markdown(
                '<div class="section">仓位与技术状态</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="grid4">

                <div class="card">
                <div class="k">建议股数</div>
                <div class="v">{shares}</div>
                <div class="s">Risk based</div>
                </div>

                <div class="card">
                <div class="k">预计风险</div>
                <div class="v">${est_risk:.0f}</div>
                <div class="s">{risk_pct*100:.1f}% limit</div>
                </div>

                <div class="card">
                <div class="k">仓位金额</div>
                <div class="v">${notional:,.0f}</div>
                <div class="s">Notional</div>
                </div>

                <div class="card">
                <div class="k">RSI / ADX</div>
                <div class="v">
                {fmt(sig['rsi'],1)} /
                {fmt(sig['adx'],1)}
                </div>
                <div class="s">Momentum / Trend</div>
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ADVANCED
            st.markdown(
                '<div class="section">V4 高级动量</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="grid4">

                <div class="card">
                <div class="k">ROC 10</div>
                <div class="v">{fmt(sig['roc10'],2)}%</div>
                <div class="s">Momentum</div>
                </div>

                <div class="card">
                <div class="k">短周期波动率</div>
                <div class="v">{fmt(sig['volatility'],1)}%</div>
                <div class="s">Volatility</div>
                </div>

                <div class="card">
                <div class="k">BB Width</div>
                <div class="v">{fmt(sig['bb_width'],3)}</div>
                <div class="s">Compression</div>
                </div>

                <div class="card">
                <div class="k">MACD H</div>
                <div class="v">{fmt(sig['macd_h'],3)}</div>
                <div class="s">Momentum</div>
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section">5分钟价格结构</div>',
                unsafe_allow_html=True
            )

            st.plotly_chart(
                make_chart(sig),
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

            st.markdown(
                '<div class="section">模型评分解释</div>',
                unsafe_allow_html=True
            )

            for reason in sig["reasons"]:

                st.markdown(
                    f'<div class="reason">{reason}</div>',
                    unsafe_allow_html=True
                )


# ============================================================
# SCANNER
# ============================================================

with tabs[1]:

    st.markdown(
        "### 🔎 V4 强信号扫描器"
    )

    watch_default = [
        "NVDA",
        "TSLA",
        "AMZN",
        "GOOGL",
        "META",
        "AMD",
        "AVGO",
        "ORCL",
        "MSFT",
        "TSM",
        "PLTR",
        "AAPL"
    ]

    watch_text = st.text_input(
        "扫描列表（逗号分隔）",
        ",".join(watch_default)
    )

    watch = [
        x.strip().upper()
        for x in watch_text.split(",")
        if x.strip()
    ][:20]

    min_score = st.slider(
        "最低显示评分",
        50,
        110,
        70,
        5
    )

    only_confirmed = st.toggle(
        "只看强确认",
        False
    )

    if st.button(
        "运行扫描器"
    ):

        rows = []
        progress = st.progress(0)

        for i, sym in enumerate(watch):

            try:

                d, provider = load_symbol(
                    sym,
                    provider_choice
                )

                daily_data = load_daily_yahoo(
                    sym
                )

                sg = (
                    evaluate(
                        d,
                        daily_data,
                        mbias
                    )
                    if d is not None
                    else None
                )

                if sg:

                    pass_filter = (
                        sg["score"] >= min_score
                    )

                    if only_confirmed:

                        pass_filter = (
                            pass_filter
                            and
                            sg["status"] == "CONFIRMED"
                        )

                    if pass_filter:

                        rows.append({
                            "symbol": sym,
                            "score": sg["score"],
                            "side": sg["side"],
                            "status": sg["status"],
                            "price": sg["entry"],
                            "long": sg["long_score"],
                            "short": sg["short_score"],
                            "gap": sg["gap"],
                            "provider": provider
                        })

            except Exception:
                pass

            progress.progress(
                (i + 1) /
                max(1, len(watch))
            )

        progress.empty()

        rows = sorted(
            rows,
            key=lambda x: (
                x["score"],
                x["gap"]
            ),
            reverse=True
        )

        if not rows:

            st.info(
                "当前没有符合过滤条件的信号。"
            )

        else:

            st.markdown("""
            <div class="rank-row rank-head">
            <div>代码</div>
            <div>评分</div>
            <div>方向</div>
            <div>状态</div>
            <div>价格</div>
            </div>
            """, unsafe_allow_html=True)

            for x in rows:

                side_text = (
                    "🟢 LONG"
                    if x["side"] == "LONG"
                    else
                    "🔴 SHORT"
                )

                status_text = {
                    "CONFIRMED": "强确认",
                    "WAIT_CONFIRM": "等待",
                    "WATCH": "观察",
                    "NO_TRADE": "禁入"
                }.get(
                    x["status"],
                    x["status"]
                )

                st.markdown(
                    f"""
                    <div class="rank-row">

                    <div>
                    <b>{x['symbol']}</b>
                    </div>

                    <div>
                    {x['score']}
                    </div>

                    <div>
                    {side_text}
                    </div>

                    <div>
                    {status_text}
                    </div>

                    <div>
                    ${x['price']:.2f}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# MARKET TAB
# ============================================================

with tabs[2]:

    st.markdown(
        "### 🌐 市场环境"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "市场偏向",
        trend_cn(mbias)
    )

    c2.metric(
        "市场评分",
        mkt_score
    )

    c3.metric(
        "VIX",
        fmt(vix, 2)
    )

    st.caption(
        mdetail
    )

    st.markdown(
        '<div class="section">V4 市场过滤逻辑</div>',
        unsafe_allow_html=True
    )

    rules = [
        "SPY、QQQ、IWM 判断价格相对 EMA20 与 VWAP 的位置。",
        "三个指数多数向上时，大盘环境偏多。",
        "三个指数多数向下时，大盘环境偏空。",
        "VIX ≥ 25 时降低风险偏好。",
        "VIX ≥ 30 时进一步降低风险偏好。",
        "VIX ≤ 17 时提高风险偏好。",
        "个股信号逆大盘方向时自动扣分。"
    ]

    for t in rules:

        st.markdown(
            f'<div class="reason">{t}</div>',
            unsafe_allow_html=True
        )


# ============================================================
# RISK TAB
# ============================================================

with tabs[3]:

    st.markdown(
        "### 🛡️ V4 风险控制"
    )

    day_loss = (
        account *
        day_loss_pct
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "单笔风险上限",
        f"${account*risk_pct:,.0f}"
    )

    c2.metric(
        "日内亏损熔断",
        f"${day_loss:,.0f}"
    )

    c3.metric(
        "最大仓位",
        f"{max_pos_pct*100:.0f}%"
    )

    st.markdown(
        '<div class="section">V4 风控规则</div>',
        unsafe_allow_html=True
    )

    risk_rules = [
        "强确认不代表必须交易，只代表模型条件高度一致。",
        "默认单笔风险为账户资金的 0.5%。",
        "默认单一标的最大仓位为账户资金的 25%。",
        "止损基于 ATR 与最近结构低点/高点共同计算。",
        "达到日内亏损熔断值后，应停止新增交易。",
        "第三推动并出现 RSI 极端值时，模型自动扣分。",
        "突破没有成交量支持时，模型识别潜在假突破并扣分。",
        "5m、15m、60m、日线四周期一致时获得额外共振评分。",
        "短周期波动率过高时自动降低信号评分。",
        "财报、CPI、FOMC、非农前后建议降低仓位。",
        "正式投入大资金前，应进行历史回测和样本外验证。"
    ]

    for t in risk_rules:

        st.markdown(
            f'<div class="reason">{t}</div>',
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="foot">

    更新
    {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    · QUANT PRO V4
    · ≥90 强确认
    · 82–89 等待
    · 70–81 观察
    · &lt;70 禁入
    · 多周期 · 动量 · 成交量 · 波动率 · 市场过滤
    · 仅用于研究与辅助判断，不构成投资建议

    </div>
    """,
    unsafe_allow_html=True
)
