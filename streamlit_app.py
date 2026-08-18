import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="美股量化系统 PRO V2", page_icon="📈", layout="wide")
st.title("📈 美股量化交易系统 PRO V2")
st.caption("5m执行｜15m确认｜60m趋势｜SPY/QQQ过滤｜量价+动能+结构+风控")

def ema(s,n): return s.ewm(span=n, adjust=False).mean()

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).rolling(n).mean(); dn=(-d.clip(upper=0)).rolling(n).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

def atr(df,n=14):
    pc=df["Close"].shift(1)
    tr=pd.concat([(df["High"]-df["Low"]),(df["High"]-pc).abs(),(df["Low"]-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vwap(df):
    tp=(df["High"]+df["Low"]+df["Close"])/3
    day=pd.Series(df.index.date,index=df.index)
    pv=tp*df["Volume"]
    return pv.groupby(day).cumsum()/df["Volume"].groupby(day).cumsum().replace(0,np.nan)

def add_indicators(df):
    x=df.copy()
    x["EMA20"]=ema(x["Close"],20); x["EMA60"]=ema(x["Close"],60)
    x["RSI"]=rsi(x["Close"]); x["ATR"]=atr(x); x["VWAP"]=vwap(x)
    macd=ema(x["Close"],12)-ema(x["Close"],26); x["MACD_H"]=macd-ema(macd,9)
    x["VOL20"]=x["Volume"].rolling(20).mean(); x["RVOL"]=x["Volume"]/x["VOL20"].replace(0,np.nan)
    x["OBV"]=(np.sign(x["Close"].diff()).fillna(0)*x["Volume"]).cumsum()
    x["HH20"]=x["High"].shift(1).rolling(20).max(); x["LL20"]=x["Low"].shift(1).rolling(20).min()
    return x

def resample_ohlcv(df,rule):
    return df.resample(rule).agg(Open=("Open","first"),High=("High","max"),Low=("Low","min"),
                                 Close=("Close","last"),Volume=("Volume","sum")).dropna()

def trend_state(df):
    if len(df)<65: return "NEUTRAL"
    r=df.iloc[-1]
    if r["Close"]>r["EMA20"]>r["EMA60"]: return "BULL"
    if r["Close"]<r["EMA20"]<r["EMA60"]: return "BEAR"
    return "NEUTRAL"

@st.cache_data(ttl=60)
def load_symbol(symbol):
    x=yf.download(symbol,period="30d",interval="5m",auto_adjust=True,prepost=False,progress=False,threads=False)
    if x is None or x.empty: return None
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    need=["Open","High","Low","Close","Volume"]
    return x[need].dropna() if all(c in x.columns for c in need) else None

def market_bias():
    out=[]
    score=0
    for sym in ["SPY","QQQ"]:
        d=load_symbol(sym)
        if d is None: continue
        f=add_indicators(d); r=f.iloc[-1]
        if r["Close"]>r["EMA20"] and r["Close"]>r["VWAP"]:
            score+=1; out.append(f"{sym}偏多")
        elif r["Close"]<r["EMA20"] and r["Close"]<r["VWAP"]:
            score-=1; out.append(f"{sym}偏空")
        else: out.append(f"{sym}中性")
    bias="BULL" if score>=1 else "BEAR" if score<=-1 else "NEUTRAL"
    return bias," / ".join(out)

def evaluate(df,mkt):
    f5=add_indicators(df)
    f15=add_indicators(resample_ohlcv(df,"15min"))
    f60=add_indicators(resample_ohlcv(df,"60min"))
    if min(len(f15),len(f60))<60: return None
    r=f5.iloc[-1]; p=f5.iloc[-2]
    t5,t15,t60=trend_state(f5),trend_state(f15),trend_state(f60)
    L=S=0; lr=[]; sr=[]

    if t60=="BULL": L+=18; lr.append("60m主趋势向上 +18")
    elif t60=="BEAR": S+=18; sr.append("60m主趋势向下 +18")
    if t15=="BULL": L+=12; lr.append("15m确认向上 +12")
    elif t15=="BEAR": S+=12; sr.append("15m确认向下 +12")

    if r["Close"]>r["EMA20"]>r["EMA60"]: L+=10; lr.append("5m EMA多头排列 +10")
    elif r["Close"]<r["EMA20"]<r["EMA60"]: S+=10; sr.append("5m EMA空头排列 +10")

    if r["Close"]>r["VWAP"]: L+=8; lr.append("价格在VWAP上方 +8")
    elif r["Close"]<r["VWAP"]: S+=8; sr.append("价格在VWAP下方 +8")

    if r["MACD_H"]>0 and r["MACD_H"]>p["MACD_H"]: L+=8; lr.append("MACD多头动能增强 +8")
    elif r["MACD_H"]<0 and r["MACD_H"]<p["MACD_H"]: S+=8; sr.append("MACD空头动能增强 +8")

    if 52<=r["RSI"]<=72: L+=7; lr.append("RSI有效多头区 +7")
    elif 28<=r["RSI"]<=48: S+=7; sr.append("RSI有效空头区 +7")

    if r["RVOL"]>=1.2:
        if r["Close"]>r["Open"]: L+=8; lr.append(f"RVOL {r['RVOL']:.2f} 放量上涨 +8")
        elif r["Close"]<r["Open"]: S+=8; sr.append(f"RVOL {r['RVOL']:.2f} 放量下跌 +8")

    if f5["OBV"].iloc[-1]>f5["OBV"].iloc[-6]: L+=4; lr.append("OBV近端上行 +4")
    elif f5["OBV"].iloc[-1]<f5["OBV"].iloc[-6]: S+=4; sr.append("OBV近端下行 +4")

    if pd.notna(r["HH20"]) and r["Close"]>r["HH20"]: L+=10; lr.append("突破20根K线前高 +10")
    if pd.notna(r["LL20"]) and r["Close"]<r["LL20"]: S+=10; sr.append("跌破20根K线前低 +10")

    if mkt=="BULL": L+=8; S-=5; lr.append("SPY/QQQ偏多 +8")
    elif mkt=="BEAR": S+=8; L-=5; sr.append("SPY/QQQ偏空 +8")

    if (t60=="BULL" and t15=="BEAR") or (t60=="BEAR" and t15=="BULL"): L-=10; S-=10

    L=max(0,int(L)); S=max(0,int(S))
    side="LONG" if L>S else "SHORT"
    score=max(L,S); gap=abs(L-S)

    if score>=85 and gap>=18: status="CONFIRMED"
    elif score>=78 and gap>=12: status="WAIT_CONFIRM"
    elif score>=65: status="WATCH"
    else: status="NO_TRADE"
    if gap<8: status="NO_TRADE"

    entry=float(r["Close"]); a=float(r["ATR"])
    if side=="LONG":
        stop=min(float(f5["Low"].tail(8).min()),entry-1.35*a); risk=entry-stop
        t1,t2,t3=entry+risk,entry+2*risk,entry+3*risk
    else:
        stop=max(float(f5["High"].tail(8).max()),entry+1.35*a); risk=stop-entry
        t1,t2,t3=entry-risk,entry-2*risk,entry-3*risk

    return dict(side=side,status=status,score=score,long_score=L,short_score=S,entry=entry,stop=stop,
                t1=t1,t2=t2,t3=t3,trend5=t5,trend15=t15,trend60=t60,
                rsi=float(r["RSI"]),rvol=float(r["RVOL"]),vwap=float(r["VWAP"]),
                reasons=(lr if side=="LONG" else sr))

def label(sig):
    if sig["status"]=="CONFIRMED": return "🟢 强做多确认" if sig["side"]=="LONG" else "🔴 强做空确认"
    if sig["status"]=="WAIT_CONFIRM": return "🟡 多头等待确认" if sig["side"]=="LONG" else "🟡 空头等待确认"
    if sig["status"]=="WATCH": return "👀 多头观察" if sig["side"]=="LONG" else "👀 空头观察"
    return "⚪ 禁止交易 / 信号冲突"

symbol=st.selectbox("选择股票",["NVDA","TSLA","AMZN","GOOGL","ORCL","META","AMD","MSFT","AAPL","TSM"])
custom=st.text_input("或者输入其他美股代码")
if custom.strip(): symbol=custom.strip().upper()

with st.expander("⚙️ 资金与风险设置"):
    account=st.number_input("账户资金 $",min_value=1000.0,value=20000.0,step=1000.0)
    risk_pct=st.slider("单笔最大风险 %",0.1,1.5,0.5,0.1)/100

if st.button("🔄 立即刷新"):
    st.cache_data.clear(); st.rerun()

with st.spinner(f"正在分析 {symbol} ..."):
    df=load_symbol(symbol); mbias,mreason=market_bias()

if df is None:
    st.error("无法获取行情，请稍后再试。"); st.stop()

sig=evaluate(df,mbias)
if sig is None:
    st.warning("数据不足。"); st.stop()

txt=label(sig)
if sig["status"]=="CONFIRMED" and sig["side"]=="LONG": st.success(f"{txt} ｜ {symbol}")
elif sig["status"]=="CONFIRMED" and sig["side"]=="SHORT": st.error(f"{txt} ｜ {symbol}")
elif sig["status"] in ["WAIT_CONFIRM","WATCH"]: st.warning(f"{txt} ｜ {symbol}")
else: st.info(f"{txt} ｜ {symbol}")

c1,c2=st.columns(2); c1.metric("信号评分",f"{sig['score']}/100"); c2.metric("当前价格",f"${sig['entry']:.2f}")
c1,c2=st.columns(2); c1.metric("多头分",sig["long_score"]); c2.metric("空头分",sig["short_score"])
st.caption(f"SPY/QQQ：{mbias} ｜ {mreason}")

st.subheader("📊 多周期")
c1,c2,c3=st.columns(3); c1.metric("5m",sig["trend5"]); c2.metric("15m",sig["trend15"]); c3.metric("60m",sig["trend60"])

st.subheader("🎯 风控")
c1,c2=st.columns(2); c1.metric("入场参考",f"${sig['entry']:.2f}"); c2.metric("结构止损",f"${sig['stop']:.2f}")
c1,c2,c3=st.columns(3); c1.metric("1R",f"${sig['t1']:.2f}"); c2.metric("2R",f"${sig['t2']:.2f}"); c3.metric("3R",f"${sig['t3']:.2f}")

per_share=abs(sig["entry"]-sig["stop"]); risk_money=account*risk_pct
shares=int(risk_money//per_share) if per_share>0 else 0
st.metric("建议股数",shares)

st.subheader("🔬 技术状态")
c1,c2=st.columns(2); c1.metric("RSI",f"{sig['rsi']:.1f}"); c2.metric("RVOL",f"{sig['rvol']:.2f}")
st.metric("VWAP",f"${sig['vwap']:.2f}")

st.subheader("🧠 信号解释")
for reason in sig["reasons"][:8]:
    st.write("•",reason)

st.subheader("📉 最近走势")
st.line_chart(df.tail(120)[["Close"]])

st.caption("V2：≥85强确认；78–84等待确认；65–77观察；<65禁止交易。")
st.caption("⚠️ 仅用于研究与辅助判断，不构成投资建议。Yahoo Finance 行情可能存在延迟。")
