import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# =========================
# 页面设置
# =========================
st.set_page_config(
    page_title="美股量化交易系统 PRO",
    page_icon="📈",
    layout="wide"
)

st.title("📈 美股量化交易系统 PRO")
st.caption("5分钟执行｜15分钟确认｜60分钟趋势｜量价 + 动能 + 风控")

# =========================
# CSS 手机优化
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.20);
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 技术指标
# =========================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    prev_close = df["Close"].shift(1)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def adx(df, period=14):

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    atr_val = atr(df, period)

    plus_di = (
        100 *
        pd.Series(plus_dm, index=df.index).rolling(period).mean()
        / atr_val
    )

    minus_di = (
        100 *
        pd.Series(minus_dm, index=df.index).rolling(period).mean()
        / atr_val
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(0, np.nan)
    )

    adx_val = dx.rolling(period).mean()

    return adx_val, plus_di, minus_di


def obv(df):

    direction = np.sign(df["Close"].diff()).fillna(0)

    return (
        direction * df["Volume"]
    ).cumsum()


def vwap(df):

    typical_price = (
        df["High"]
        + df["Low"]
        + df["Close"]
    ) / 3

    date_index = pd.Series(
        df.index.date,
        index=df.index
    )

    pv = typical_price * df["Volume"]

    cumulative_pv = pv.groupby(date_index).cumsum()

    cumulative_volume = (
        df["Volume"]
        .groupby(date_index)
        .cumsum()
        .replace(0, np.nan)
    )

    return cumulative_pv / cumulative_volume


# =========================
# 添加技术指标
# =========================

def add_indicators(df):

    x = df.copy()

    x["EMA20"] = ema(x["Close"], 20)
    x["EMA60"] = ema(x["Close"], 60)

    x["RSI"] = rsi(x["Close"])

    x["ATR"] = atr(x)

    x["VWAP"] = vwap(x)

    fast = ema(x["Close"], 12)
    slow = ema(x["Close"], 26)

    x["MACD"] = fast - slow
    x["MACD_SIGNAL"] = ema(x["MACD"], 9)
    x["MACD_HIST"] = (
        x["MACD"] - x["MACD_SIGNAL"]
    )

    adx_val, plus_di, minus_di = adx(x)

    x["ADX"] = adx_val
    x["PLUS_DI"] = plus_di
    x["MINUS_DI"] = minus_di

    x["OBV"] = obv(x)

    x["VOL_AVG20"] = (
        x["Volume"]
        .rolling(20)
        .mean()
    )

    x["RVOL"] = (
        x["Volume"]
        /
        x["VOL_AVG20"]
        .replace(0, np.nan)
    )

    x["HIGH20"] = (
        x["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    x["LOW20"] = (
        x["Low"]
        .shift(1)
        .rolling(20)
        .min()
    )

    return x


# =========================
# 周期重采样
# =========================

def resample_data(df, interval):

    rule = {
        "15m": "15min",
        "60m": "60min"
    }[interval]

    x = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    return x.dropna()


# =========================
# 趋势判断
# =========================

def trend_state(df):

    if len(df) < 70:
        return "NEUTRAL"

    r = df.iloc[-1]

    bull = (
        r["Close"] > r["EMA20"]
        and
        r["EMA20"] > r["EMA60"]
    )

    bear = (
        r["Close"] < r["EMA20"]
        and
        r["EMA20"] < r["EMA60"]
    )

    if bull:
        return "BULL"

    if bear:
        return "BEAR"

    return "NEUTRAL"


# =========================
# 当前推动腿
# =========================

def detect_leg(df, side):

    recent = df.tail(30)

    if side == "LONG":

        rolling_high = (
            recent["High"]
            .rolling(6)
            .max()
        )

        pushes = (
            recent["High"] >= rolling_high
        ).sum()

    else:

        rolling_low = (
            recent["Low"]
            .rolling(6)
            .min()
        )

        pushes = (
            recent["Low"] <= rolling_low
        ).sum()

    if pushes >= 9:
        return "THIRD_PUSH"

    if pushes >= 6:
        return "SECOND_PUSH"

    return "FIRST_PUSH"


# =========================
# 行情下载
# =========================

@st.cache_data(ttl=60)
def get_market_data(symbol):

    data = yf.download(
        symbol,
        period="60d",
        interval="5m",
        progress=False,
        auto_adjust=True,
        prepost=False
    )

    if data is None or len(data) == 0:
        return None

    # yfinance 有时返回 MultiIndex
    if isinstance(data.columns, pd.MultiIndex):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    data = data[required]

    data = data.dropna()

    return data


# =========================
# 核心量化评分
# =========================

def calculate_signal(df5):

    if df5 is None or len(df5) < 150:
        return None

    f5 = add_indicators(df5)

    df15 = resample_data(df5, "15m")
    df60 = resample_data(df5, "60m")

    f15 = add_indicators(df15)
    f60 = add_indicators(df60)

    if len(f15) < 70 or len(f60) < 70:
        return None

    row = f5.iloc[-1]
    prev = f5.iloc[-2]

    trend5 = trend_state(f5)
    trend15 = trend_state(f15)
    trend60 = trend_state(f60)

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    # =====================
    # 60分钟趋势 20分
    # =====================

    if trend60 == "BULL":

        long_score += 20
        long_reasons.append(
            "60分钟主趋势向上"
        )

    elif trend60 == "BEAR":

        short_score += 20
        short_reasons.append(
            "60分钟主趋势向下"
        )

    # =====================
    # 15分钟趋势 15分
    # =====================

    if trend15 == "BULL":

        long_score += 15
        long_reasons.append(
            "15分钟趋势向上"
        )

    elif trend15 == "BEAR":

        short_score += 15
        short_reasons.append(
            "15分钟趋势向下"
        )

    # =====================
    # 5分钟 EMA 15分
    # =====================

    if (
        row["Close"] > row["EMA20"]
        and
        row["EMA20"] > row["EMA60"]
    ):

        long_score += 15

        long_reasons.append(
            "5分钟 EMA20 > EMA60"
        )

    elif (
        row["Close"] < row["EMA20"]
        and
        row["EMA20"] < row["EMA60"]
    ):

        short_score += 15

        short_reasons.append(
            "5分钟 EMA20 < EMA60"
        )

    # =====================
    # VWAP 10分
    # =====================

    if row["Close"] > row["VWAP"]:

        long_score += 10

        long_reasons.append(
            "价格位于 VWAP 上方"
        )

    else:

        short_score += 10

        short_reasons.append(
            "价格位于 VWAP 下方"
        )

    # =====================
    # MACD 10分
    # =====================

    if (
        row["MACD_HIST"] > 0
        and
        row["MACD_HIST"]
        >
        prev["MACD_HIST"]
    ):

        long_score += 10

        long_reasons.append(
            "MACD 多头动能增强"
        )

    elif (
        row["MACD_HIST"] < 0
        and
        row["MACD_HIST"]
        <
        prev["MACD_HIST"]
    ):

        short_score += 10

        short_reasons.append(
            "MACD 空头动能增强"
        )

    # =====================
    # RSI 10分
    # =====================

    if (
        row["RSI"] >= 52
        and
        row["RSI"] <= 72
    ):

        long_score += 10

        long_reasons.append(
            "RSI 位于有效多头区域"
        )

    elif (
        row["RSI"] >= 28
        and
        row["RSI"] <= 48
    ):

        short_score += 10

        short_reasons.append(
            "RSI 位于有效空头区域"
        )

    # =====================
    # RVOL 成交量 10分
    # =====================

    if row["RVOL"] >= 1.2:

        if row["Close"] > row["Open"]:

            long_score += 10

            long_reasons.append(
                f"放量上涨 RVOL {row['RVOL']:.2f}"
            )

        else:

            short_score += 10

            short_reasons.append(
                f"放量下跌 RVOL {row['RVOL']:.2f}"
            )

    # =====================
    # OBV 5分
    # =====================

    obv_now = f5["OBV"].iloc[-1]
    obv_old = f5["OBV"].iloc[-6]

    if obv_now > obv_old:

        long_score += 5

        long_reasons.append(
            "OBV 资金流向上"
        )

    elif obv_now < obv_old:

        short_score += 5

        short_reasons.append(
            "OBV 资金流向下"
        )

    # =====================
    # 突破 10分
    # =====================

    if row["Close"] > row["HIGH20"]:

        long_score += 10

        long_reasons.append(
            "突破最近20根K线高点"
        )

    elif row["Close"] < row["LOW20"]:

        short_score += 10

        short_reasons.append(
            "跌破最近20根K线低点"
        )

    # =====================
    # ADX 趋势强度 5分
    # =====================

    if row["ADX"] >= 25:

        if row["PLUS_DI"] > row["MINUS_DI"]:

            long_score += 5

            long_reasons.append(
                "ADX趋势强且 +DI 占优"
            )

        else:

            short_score += 5

            short_reasons.append(
                "ADX趋势强且 -DI 占优"
            )

    # =====================
    # 周期冲突过滤
    # =====================

    if (
        trend60 == "BULL"
        and
        trend15 == "BEAR"
    ):

        long_score -= 10
        short_score -= 10

    if (
        trend60 == "BEAR"
        and
        trend15 == "BULL"
    ):

        long_score -= 10
        short_score -= 10

    # =====================
    # 决定方向
    # =====================

    if max(long_score, short_score) < 65:

        side = "WAIT"

    elif long_score > short_score:

        side = "LONG"

    else:

        side = "SHORT"

    score = max(
        long_score,
        short_score
    )

    reasons = (
        long_reasons
        if side == "LONG"
        else short_reasons
    )

    # =====================
    # 风控
    # =====================

    entry = float(row["Close"])
    atr_value = float(row["ATR"])

    if side == "LONG":

        recent_low = float(
            f5["Low"]
            .tail(8)
            .min()
        )

        atr_stop = (
            entry
            -
            1.35 * atr_value
        )

        stop = min(
            recent_low,
            atr_stop
        )

        risk = (
            entry
            -
            stop
        )

        target1 = (
            entry
            +
            risk
        )

        target2 = (
            entry
            +
            risk * 2
        )

        target3 = (
            entry
            +
            risk * 3
        )

    elif side == "SHORT":

        recent_high = float(
            f5["High"]
            .tail(8)
            .max()
        )

        atr_stop = (
            entry
            +
            1.35 * atr_value
        )

        stop = max(
            recent_high,
            atr_stop
        )

        risk = (
            stop
            -
            entry
        )

        target1 = (
            entry
            -
            risk
        )

        target2 = (
            entry
            -
            risk * 2
        )

        target3 = (
            entry
            -
            risk * 3
        )

    else:

        stop = np.nan
        target1 = np.nan
        target2 = np.nan
        target3 = np.nan

    # =====================
    # 市场环境
    # =====================

    if (
        trend60 == "BULL"
        and
        trend15 == "BULL"
    ):

        regime = "STRONG_UPTREND"

    elif (
        trend60 == "BEAR"
        and
        trend15 == "BEAR"
    ):

        regime = "STRONG_DOWNTREND"

    else:

        regime = "TRANSITION"

    # =====================
    # 强度
    # =====================

    if score >= 85:

        strength = "VERY_STRONG"

    elif score >= 75:

        strength = "STRONG"

    elif score >= 65:

        strength = "NORMAL"

    else:

        strength = "WEAK"

    leg = detect_leg(
        f5,
        side
        if side != "WAIT"
        else "LONG"
    )

    return {

        "side": side,

        "score": int(score),

        "entry": entry,

        "stop": stop,

        "target1": target1,

        "target2": target2,

        "target3": target3,

        "trend5": trend5,

        "trend15": trend15,

        "trend60": trend60,

        "regime": regime,

        "strength": strength,

        "leg": leg,

        "rsi": row["RSI"],

        "adx": row["ADX"],

        "rvol": row["RVOL"],

        "atr": row["ATR"],

        "vwap": row["VWAP"],

        "reasons": reasons
    }


# =========================
# 股票选择
# =========================

default_symbols = [
    "NVDA",
    "TSLA",
    "AMZN",
    "GOOGL",
    "ORCL",
    "META",
    "AMD",
    "MSFT",
    "AAPL",
    "TSM"
]

symbol = st.selectbox(
    "选择股票",
    default_symbols
)

custom = st.text_input(
    "或者输入其他美股代码"
)

if custom:

    symbol = (
        custom
        .upper()
        .strip()
    )


# =========================
# 风险参数
# =========================

with st.expander(
    "⚙️ 资金与风险设置"
):

    account = st.number_input(
        "账户资金 $",
        min_value=1000.0,
        value=20000.0,
        step=1000.0
    )

    risk_pct = (
        st.slider(
            "单笔最大风险 %",
            0.1,
            2.0,
            0.5,
            0.1
        )
        / 100
    )


# =========================
# 获取数据
# =========================

with st.spinner(
    f"正在分析 {symbol} ..."
):

    df = get_market_data(
        symbol
    )


if df is None:

    st.error(
        "无法获取行情，请检查股票代码。"
    )

    st.stop()


signal = calculate_signal(
    df
)


if signal is None:

    st.warning(
        "历史数据不足，暂时无法计算。"
    )

    st.stop()


# =========================
# 主信号显示
# =========================

side = signal["side"]


if side == "LONG":

    st.success(
        f"🟢 做多确认 | {symbol}"
    )

elif side == "SHORT":

    st.error(
        f"🔴 做空确认 | {symbol}"
    )

else:

    st.warning(
        f"🟡 暂不交易 | {symbol}"
    )


# =========================
# 第一排
# =========================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "信号评分",
    f"{signal['score']}/100"
)

c2.metric(
    "当前价格",
    f"${signal['entry']:.2f}"
)

c3.metric(
    "信号强度",
    signal["strength"]
)

c4.metric(
    "市场环境",
    signal["regime"]
)


# =========================
# 周期
# =========================

st.subheader(
    "📊 多周期趋势"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "5分钟",
    signal["trend5"]
)

c2.metric(
    "15分钟",
    signal["trend15"]
)

c3.metric(
    "60分钟",
    signal["trend60"]
)


# =========================
# 风控
# =========================

if side != "WAIT":

    st.subheader(
        "🎯 入场与风险控制"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "入场参考",
        f"${signal['entry']:.2f}"
    )

    c2.metric(
        "结构止损",
        f"${signal['stop']:.2f}"
    )

    c3.metric(
        "1R",
        f"${signal['target1']:.2f}"
    )

    c4.metric(
        "2R",
        f"${signal['target2']:.2f}"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "3R",
        f"${signal['target3']:.2f}"
    )

    c2.metric(
        "当前推动",
        signal["leg"]
    )


# =========================
# 自动仓位
# =========================

if side != "WAIT":

    per_share_risk = abs(
        signal["entry"]
        -
        signal["stop"]
    )

    risk_money = (
        account
        *
        risk_pct
    )

    shares_by_risk = int(
        risk_money
        /
        per_share_risk
    )

    max_notional = (
        account
        *
        0.25
    )

    shares_by_notional = int(
        max_notional
        /
        signal["entry"]
    )

    shares = max(
        0,
        min(
            shares_by_risk,
            shares_by_notional
        )
    )

    st.subheader(
        "💰 自动仓位管理"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "建议股数",
        shares
    )

    c2.metric(
        "预计风险",
        f"${shares * per_share_risk:.2f}"
    )

    c3.metric(
        "预计仓位",
        f"${shares * signal['entry']:.2f}"
    )


# =========================
# 技术状态
# =========================

st.subheader(
    "🔬 技术状态"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "RSI",
    f"{signal['rsi']:.1f}"
)

c2.metric(
    "ADX",
    f"{signal['adx']:.1f}"
)

c3.metric(
    "RVOL",
    f"{signal['rvol']:.2f}"
)

c4.metric(
    "VWAP",
    f"${signal['vwap']:.2f}"
)


# =========================
# 信号原因
# =========================

st.subheader(
    "🧠 信号解释"
)

if len(signal["reasons"]) == 0:

    st.write(
        "当前没有足够强的确认条件。"
    )

else:

    for reason in signal["reasons"]:

        st.write(
            "•",
            reason
        )


# =========================
# K线趋势图
# =========================

st.subheader(
    "📉 最近走势"
)

chart_df = df.tail(
    100
)[["Close"]]

st.line_chart(
    chart_df
)


# =========================
# 更新时间
# =========================

st.caption(
    f"最后刷新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

st.caption(
    "⚠️ 量化模型仅用于研究和辅助判断，不构成投资建议。Yahoo Finance 行情可能存在延迟。"
)
