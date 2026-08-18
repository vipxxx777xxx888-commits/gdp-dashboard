import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Quant PRO V2.1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# PROFESSIONAL MOBILE-FIRST UI
# =========================================================
st.markdown("""
<style>
:root{
  --bg:#0b0f14;
  --panel:#111820;
  --panel2:#151d27;
  --line:#233041;
  --text:#e8eef6;
  --muted:#8fa1b5;
  --green:#35d07f;
  --red:#ff5f6d;
  --yellow:#f5c451;
  --blue:#5aa7ff;
}

html, body, [class*="css"]  {
  font-size: 13px;
}

.stApp{
  background: linear-gradient(180deg,#0a0e13 0%, #0d1218 100%);
  color: var(--text);
}

.block-container{
  padding-top: .75rem !important;
  padding-bottom: 1.4rem !important;
  max-width: 1180px !important;
}

h1{font-size:1.28rem !important; margin-bottom:.15rem !important; letter-spacing:.2px;}
h2{font-size:1rem !important; margin-top:.7rem !important; margin-bottom:.4rem !important;}
h3{font-size:.92rem !important;}
p, label, .stCaption{font-size:.78rem !important;}

div[data-testid="stMetric"]{
  background: linear-gradient(180deg,#121a23,#10171f);
  border: 1px solid var(--line);
  padding: 9px 10px;
  border-radius: 10px;
  min-height: 76px;
}
div[data-testid="stMetric"] label{
  color:var(--muted) !important;
  font-size:.68rem !important;
}
div[data-testid="stMetricValue"]{
  font-size:1.18rem !important;
  line-height:1.15 !important;
}
div[data-testid="stMetricDelta"]{
  font-size:.68rem !important;
}

.stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label{
  font-size:.72rem !important;
  color:var(--muted) !important;
}

.stButton>button{
  width:100%;
  border-radius:9px;
  min-height:36px;
  border:1px solid var(--line);
  background:#121a23;
  color:var(--text);
  font-size:.78rem;
}

div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input{
  background:#10171f !important;
  border-color:var(--line) !important;
  min-height:38px !important;
  font-size:.82rem !important;
}

.q-header{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:10px;
  padding:2px 0 8px 0;
}
.q-title{font-weight:800;font-size:1.15rem;letter-spacing:.15px}
.q-sub{color:var(--muted);font-size:.69rem;margin-top:2px}
.q-badge{
  display:inline-flex;align-items:center;gap:6px;
  padding:5px 8px;border:1px solid var(--line);
  background:#10171f;border-radius:999px;color:var(--muted);
  font-size:.66rem;white-space:nowrap
}

.signal{
  border-radius:12px;
  padding:12px 13px;
  margin:6px 0 10px 0;
  border:1px solid var(--line);
  background:#111820;
}
.signal .big{font-size:1.08rem;font-weight:850}
.signal .small{font-size:.69rem;color:var(--muted);margin-top:3px}
.signal.long{border-left:4px solid var(--green)}
.signal.short{border-left:4px solid var(--red)}
.signal.wait{border-left:4px solid var(--yellow)}
.signal.none{border-left:4px solid #718096}

.section-title{
  font-size:.76rem;
  color:#b9c7d6;
  font-weight:750;
  letter-spacing:.15px;
  margin:10px 0 5px 1px;
}
.mini-grid{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:7px;
}
.mini-card{
  background:#111820;
  border:1px solid var(--line);
  border-radius:9px;
  padding:8px 9px;
}
.mini-card .k{font-size:.63rem;color:var(--muted);margin-bottom:3px}
.mini-card .v{font-size:.88rem;font-weight:750}
.mini-card .s{font-size:.61rem;color:var(--muted);margin-top:2px}
.green{color:var(--green)}
.red{color:var(--red)}
.yellow{color:var(--yellow)}
.blue{color:var(--blue)}
.muted{color:var(--muted)}

.reason-box{
  background:#10171f;
  border:1px solid var(--line);
  border-radius:10px;
  padding:8px 10px;
  margin-bottom:5px;
  font-size:.72rem;
}
.footer-note{
  color:var(--muted);
  font-size:.62rem;
  text-align:center;
  margin-top:12px;
}

@media (max-width: 700px){
  .block-container{padding-left:.7rem !important;padding-right:.7rem !important;}
  .mini-grid{grid-template-columns:repeat(2,1fr);}
  div[data-testid="stMetric"]{min-height:70px;padding:8px 9px;}
  div[data-testid="stMetricValue"]{font-size:1.02rem !important;}
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# INDICATORS
# =========================================================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    plus_di = 100 * plus_dm.rolling(n).mean() / a.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(n).mean() / a.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean(), plus_di, minus_di

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = pd.Series(df.index.date, index=df.index)
    pv = tp * df["Volume"]
    return pv.groupby(day).cumsum() / df["Volume"].groupby(day).cumsum().replace(0, np.nan)

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
    x["OBV"] = (np.sign(x["Close"].diff()).fillna(0) * x["Volume"]).cumsum()
    x["HH20"] = x["High"].shift(1).rolling(20).max()
    x["LL20"] = x["Low"].shift(1).rolling(20).min()
    return x

def resample_ohlcv(df, rule):
    return df.resample(rule).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum")
    ).dropna()

def trend_state(df):
    if len(df) < 65:
        return "NEUTRAL"
    r = df.iloc[-1]
    if r["Close"] > r["EMA20"] > r["EMA60"]:
        return "BULL"
    if r["Close"] < r["EMA20"] < r["EMA60"]:
        return "BEAR"
    return "NEUTRAL"

@st.cache_data(ttl=60)
def load_symbol(symbol):
    x = yf.download(
        symbol,
        period="30d",
        interval="5m",
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

def market_bias():
    score = 0
    detail = []
    for sym in ["SPY", "QQQ"]:
        d = load_symbol(sym)
        if d is None:
            detail.append(f"{sym} 无数据")
            continue
        f = add_indicators(d)
        r = f.iloc[-1]
        if r["Close"] > r["EMA20"] and r["Close"] > r["VWAP"]:
            score += 1
            detail.append(f"{sym} 偏多")
        elif r["Close"] < r["EMA20"] and r["Close"] < r["VWAP"]:
            score -= 1
            detail.append(f"{sym} 偏空")
        else:
            detail.append(f"{sym} 中性")
    bias = "BULL" if score >= 1 else "BEAR" if score <= -1 else "NEUTRAL"
    return bias, " / ".join(detail)

def detect_leg(df, side):
    recent = df.tail(30)
    if side == "LONG":
        pushes = (recent["High"] >= recent["High"].rolling(6).max()).sum()
    else:
        pushes = (recent["Low"] <= recent["Low"].rolling(6).min()).sum()
    if pushes >= 9:
        return "THIRD_PUSH"
    if pushes >= 6:
        return "SECOND_PUSH"
    return "FIRST_PUSH"

def evaluate(df, mkt):
    if df is None or len(df) < 200:
        return None

    f5 = add_indicators(df)
    f15 = add_indicators(resample_ohlcv(df, "15min"))
    f60 = add_indicators(resample_ohlcv(df, "60min"))
    if min(len(f15), len(f60)) < 60:
        return None

    r = f5.iloc[-1]
    p = f5.iloc[-2]
    t5, t15, t60 = trend_state(f5), trend_state(f15), trend_state(f60)

    L, S = 0, 0
    lr, sr = [], []

    if t60 == "BULL":
        L += 18; lr.append("60m 主趋势向上 +18")
    elif t60 == "BEAR":
        S += 18; sr.append("60m 主趋势向下 +18")

    if t15 == "BULL":
        L += 12; lr.append("15m 趋势确认向上 +12")
    elif t15 == "BEAR":
        S += 12; sr.append("15m 趋势确认向下 +12")

    if r["Close"] > r["EMA20"] > r["EMA60"]:
        L += 10; lr.append("5m EMA 多头排列 +10")
    elif r["Close"] < r["EMA20"] < r["EMA60"]:
        S += 10; sr.append("5m EMA 空头排列 +10")

    if r["Close"] > r["VWAP"]:
        L += 8; lr.append("价格位于 VWAP 上方 +8")
    elif r["Close"] < r["VWAP"]:
        S += 8; sr.append("价格位于 VWAP 下方 +8")

    if r["MACD_H"] > 0 and r["MACD_H"] > p["MACD_H"]:
        L += 8; lr.append("MACD 多头动能增强 +8")
    elif r["MACD_H"] < 0 and r["MACD_H"] < p["MACD_H"]:
        S += 8; sr.append("MACD 空头动能增强 +8")

    if 52 <= r["RSI"] <= 72:
        L += 7; lr.append("RSI 位于有效多头区 +7")
    elif 28 <= r["RSI"] <= 48:
        S += 7; sr.append("RSI 位于有效空头区 +7")

    if pd.notna(r["RVOL"]) and r["RVOL"] >= 1.2:
        if r["Close"] > r["Open"]:
            L += 8; lr.append(f"RVOL {r['RVOL']:.2f} 放量上涨 +8")
        elif r["Close"] < r["Open"]:
            S += 8; sr.append(f"RVOL {r['RVOL']:.2f} 放量下跌 +8")

    if f5["OBV"].iloc[-1] > f5["OBV"].iloc[-6]:
        L += 4; lr.append("OBV 近端上行 +4")
    elif f5["OBV"].iloc[-1] < f5["OBV"].iloc[-6]:
        S += 4; sr.append("OBV 近端下行 +4")

    if pd.notna(r["HH20"]) and r["Close"] > r["HH20"]:
        L += 10; lr.append("突破 20 根 K 线前高 +10")
    if pd.notna(r["LL20"]) and r["Close"] < r["LL20"]:
        S += 10; sr.append("跌破 20 根 K 线前低 +10")

    if pd.notna(r["ADX"]) and r["ADX"] >= 25:
        if r["PLUS_DI"] > r["MINUS_DI"]:
            L += 5; lr.append("ADX>25 且 +DI 占优 +5")
        else:
            S += 5; sr.append("ADX>25 且 -DI 占优 +5")

    if mkt == "BULL":
        L += 8; S -= 5; lr.append("SPY/QQQ 大盘偏多 +8")
    elif mkt == "BEAR":
        S += 8; L -= 5; sr.append("SPY/QQQ 大盘偏空 +8")

    if (t60 == "BULL" and t15 == "BEAR") or (t60 == "BEAR" and t15 == "BULL"):
        L -= 10
        S -= 10

    L = max(0, int(L))
    S = max(0, int(S))
    side = "LONG" if L > S else "SHORT"
    score = max(L, S)
    gap = abs(L - S)

    if score >= 85 and gap >= 18:
        status = "CONFIRMED"
    elif score >= 78 and gap >= 12:
        status = "WAIT_CONFIRM"
    elif score >= 65:
        status = "WATCH"
    else:
        status = "NO_TRADE"

    if gap < 8:
        status = "NO_TRADE"

    entry = float(r["Close"])
    a = float(r["ATR"]) if pd.notna(r["ATR"]) else np.nan

    stop = t1 = t2 = t3 = np.nan
    if np.isfinite(a) and a > 0:
        if side == "LONG":
            stop = min(float(f5["Low"].tail(8).min()), entry - 1.35 * a)
            risk = entry - stop
            t1, t2, t3 = entry + risk, entry + 2*risk, entry + 3*risk
        else:
            stop = max(float(f5["High"].tail(8).max()), entry + 1.35 * a)
            risk = stop - entry
            t1, t2, t3 = entry - risk, entry - 2*risk, entry - 3*risk

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
        "rsi": float(r["RSI"]) if pd.notna(r["RSI"]) else np.nan,
        "adx": float(r["ADX"]) if pd.notna(r["ADX"]) else np.nan,
        "rvol": float(r["RVOL"]) if pd.notna(r["RVOL"]) else np.nan,
        "vwap": float(r["VWAP"]) if pd.notna(r["VWAP"]) else np.nan,
        "atr": a,
        "leg": detect_leg(f5, side),
        "reasons": (lr if side == "LONG" else sr)[:8],
        "chart": f5.tail(120)[["Close", "EMA20", "EMA60", "VWAP"]]
    }

def signal_label(sig):
    if sig["status"] == "CONFIRMED":
        return ("强做多确认", "long") if sig["side"] == "LONG" else ("强做空确认", "short")
    if sig["status"] == "WAIT_CONFIRM":
        return ("多头等待确认", "wait") if sig["side"] == "LONG" else ("空头等待确认", "wait")
    if sig["status"] == "WATCH":
        return ("多头观察", "wait") if sig["side"] == "LONG" else ("空头观察", "wait")
    return ("禁止交易 / 信号冲突", "none")

def fmt(x, digits=2):
    return "-" if not np.isfinite(x) else f"{x:.{digits}f}"

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="q-header">
  <div>
    <div class="q-title">📈 QUANT PRO V2.1</div>
    <div class="q-sub">Multi-Timeframe Signal Engine · Mobile Dashboard</div>
  </div>
  <div class="q-badge">LIVE · 5m</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CONTROLS
# =========================================================
c1, c2, c3 = st.columns([1.15, 1.35, .55])

with c1:
    symbol = st.selectbox(
        "标的",
        ["NVDA","TSLA","AMZN","GOOGL","ORCL","META","AMD","MSFT","AAPL","TSM"],
        index=4
    )
with c2:
    custom = st.text_input("自定义代码", placeholder="例如 AVGO")
    if custom.strip():
        symbol = custom.strip().upper()
with c3:
    st.write("")
    st.write("")
    if st.button("刷新"):
        st.cache_data.clear()
        st.rerun()

with st.expander("资金与风险设置", expanded=False):
    a1, a2, a3 = st.columns(3)
    with a1:
        account = st.number_input("账户资金 $", min_value=1000.0, value=20000.0, step=1000.0)
    with a2:
        risk_pct = st.slider("单笔风险 %", 0.1, 1.5, 0.5, 0.1) / 100
    with a3:
        max_pos_pct = st.slider("最大仓位 %", 5, 50, 25, 5) / 100

# =========================================================
# DATA
# =========================================================
with st.spinner(f"分析 {symbol} ..."):
    df = load_symbol(symbol)
    mbias, mreason = market_bias()

if df is None:
    st.error("无法获取行情，请稍后重试。")
    st.stop()

sig = evaluate(df, mbias)
if sig is None:
    st.warning("数据不足，暂时无法计算。")
    st.stop()

label, cls = signal_label(sig)

# =========================================================
# TOP SIGNAL CARD
# =========================================================
side_cn = "多头" if sig["side"] == "LONG" else "空头"
st.markdown(f"""
<div class="signal {cls}">
  <div class="big">{label} · {symbol}</div>
  <div class="small">评分 {sig['score']}/100 · {side_cn}领先 {sig['gap']} 分 · SPY/QQQ {mbias}</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# COMPACT SUMMARY
# =========================================================
st.markdown('<div class="section-title">核心概览</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="mini-grid">
  <div class="mini-card"><div class="k">当前价格</div><div class="v">${fmt(sig['entry'])}</div><div class="s">Latest 5m</div></div>
  <div class="mini-card"><div class="k">信号评分</div><div class="v">{sig['score']}/100</div><div class="s">{sig['status']}</div></div>
  <div class="mini-card"><div class="k">多头 / 空头</div><div class="v"><span class="green">{sig['long_score']}</span> / <span class="red">{sig['short_score']}</span></div><div class="s">Score split</div></div>
  <div class="mini-card"><div class="k">推动阶段</div><div class="v">{sig['leg']}</div><div class="s">Structure</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">多周期共振</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="mini-grid">
  <div class="mini-card"><div class="k">5分钟</div><div class="v">{sig['trend5']}</div><div class="s">Execution</div></div>
  <div class="mini-card"><div class="k">15分钟</div><div class="v">{sig['trend15']}</div><div class="s">Confirmation</div></div>
  <div class="mini-card"><div class="k">60分钟</div><div class="v">{sig['trend60']}</div><div class="s">Primary trend</div></div>
  <div class="mini-card"><div class="k">大盘环境</div><div class="v">{mbias}</div><div class="s">{mreason}</div></div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# RISK LEVELS
# =========================================================
st.markdown('<div class="section-title">交易结构</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="mini-grid">
  <div class="mini-card"><div class="k">入场参考</div><div class="v">${fmt(sig['entry'])}</div></div>
  <div class="mini-card"><div class="k">结构止损</div><div class="v red">${fmt(sig['stop'])}</div></div>
  <div class="mini-card"><div class="k">1R</div><div class="v">${fmt(sig['t1'])}</div></div>
  <div class="mini-card"><div class="k">2R</div><div class="v">${fmt(sig['t2'])}</div></div>
  <div class="mini-card"><div class="k">3R</div><div class="v">${fmt(sig['t3'])}</div></div>
  <div class="mini-card"><div class="k">ATR</div><div class="v">{fmt(sig['atr'])}</div></div>
  <div class="mini-card"><div class="k">VWAP</div><div class="v">${fmt(sig['vwap'])}</div></div>
  <div class="mini-card"><div class="k">RVOL</div><div class="v">{fmt(sig['rvol'])}</div></div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# POSITION SIZE
# =========================================================
per_share = abs(sig["entry"] - sig["stop"]) if np.isfinite(sig["stop"]) else np.nan
risk_money = account * risk_pct

if np.isfinite(per_share) and per_share > 0:
    by_risk = int(risk_money // per_share)
    by_notional = int((account * max_pos_pct) // sig["entry"])
    shares = max(0, min(by_risk, by_notional))
    est_risk = shares * per_share
    notional = shares * sig["entry"]
else:
    shares = 0
    est_risk = 0
    notional = 0

st.markdown('<div class="section-title">仓位管理</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="mini-grid">
  <div class="mini-card"><div class="k">建议股数</div><div class="v">{shares}</div><div class="s">Risk based</div></div>
  <div class="mini-card"><div class="k">预计风险</div><div class="v">${est_risk:.0f}</div><div class="s">{risk_pct*100:.1f}% limit</div></div>
  <div class="mini-card"><div class="k">预计仓位</div><div class="v">${notional:.0f}</div><div class="s">{(notional/account*100 if account else 0):.1f}% equity</div></div>
  <div class="mini-card"><div class="k">RSI / ADX</div><div class="v">{fmt(sig['rsi'],1)} / {fmt(sig['adx'],1)}</div><div class="s">Momentum / trend</div></div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CHART
# =========================================================
st.markdown('<div class="section-title">价格结构 · 最近120根5分钟K</div>', unsafe_allow_html=True)
st.line_chart(sig["chart"], height=250)

# =========================================================
# EXPLANATION
# =========================================================
st.markdown('<div class="section-title">模型解释</div>', unsafe_allow_html=True)
for reason in sig["reasons"]:
    st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="footer-note">
更新：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ·
V2.1规则：≥85强确认 · 78–84等待 · 65–77观察 · &lt;65禁入<br>
仅用于研究与辅助判断；Yahoo Finance 行情可能延迟。
</div>
""", unsafe_allow_html=True)
