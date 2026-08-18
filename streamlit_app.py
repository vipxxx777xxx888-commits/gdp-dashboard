import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="QUANT PRO 4.1", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

TECH100 = [
"NVDA","MSFT","AAPL","AMZN","GOOGL","META","AVGO","TSLA","AMD","ORCL",
"PLTR","NFLX","TSM","MU","ARM","QCOM","INTC","MRVL","SMCI","DELL",
"ANET","CRWD","PANW","FTNT","ZS","NET","DDOG","MDB","SNOW","NOW",
"CRM","ADBE","INTU","SHOP","UBER","ABNB","DASH","COIN","HOOD","SOFI",
"APP","RBLX","TTD","ROKU","SPOT","PINS","SNAP","RDDT","DUOL","MELI",
"ASML","AMAT","LRCX","KLAC","TER","MCHP","ON","NXPI","ADI","TXN",
"MPWR","SWKS","QRVO","WDC","STX","PSTG","NTAP","CIEN","LITE","COHR",
"PATH","AI","SOUN","BBAI","IONQ","RGTI","QBTS","QUBT","RKLB","ASTS",
"JOBY","ACHR","SERV","SYM","ISRG","ROK","CGNX","AZTA","VRT","ETN",
"APH","GLW","KEYS","CDNS","SNPS","ADSK","TEAM","HUBS","OKTA","GTLB"
]

st.markdown("""
<style>
:root{--bg:#070a0f;--panel:#0d131c;--panel2:#111a25;--line:#1d2a3a;--text:#edf4fb;--muted:#7f93a8;--green:#2ed47a;--red:#ff5f6d;--yellow:#f3c84b;--blue:#4da3ff}
.stApp{background:linear-gradient(180deg,#06090d,#0a1018);color:var(--text)}
.block-container{max-width:1180px;padding:.5rem .65rem 2rem!important}
header[data-testid="stHeader"]{background:transparent}
h1,h2,h3{letter-spacing:-.02em}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:8px 2px 12px}
.brand{font-size:1.25rem;font-weight:900}.brand span{color:var(--blue)}
.sub{font-size:.68rem;color:var(--muted)}
.badge{font-size:.62rem;padding:5px 8px;border:1px solid var(--line);border-radius:999px;background:#0b121b}
.hero{border:1px solid var(--line);border-radius:16px;padding:14px;background:linear-gradient(135deg,#101a26,#0b1119);margin-bottom:9px}
.hero-row{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}
.ticker{font-size:1.25rem;font-weight:900}.price{font-size:1.7rem;font-weight:900;margin-top:2px}
.signal{font-size:1rem;font-weight:900;text-align:right}.score{font-size:2rem;font-weight:950;text-align:right}
.muted{color:var(--muted);font-size:.65rem}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:7px 0}
.card{border:1px solid var(--line);border-radius:11px;background:var(--panel);padding:9px}
.k{font-size:.58rem;color:var(--muted)}.v{font-size:.88rem;font-weight:850;margin-top:2px}
.green{color:var(--green)}.red{color:var(--red)}.yellow{color:var(--yellow)}
.tf{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:8px}.tf div{text-align:center;border:1px solid var(--line);border-radius:9px;padding:7px;background:#0b121b}
.plan{display:grid;grid-template-columns:repeat(5,1fr);gap:5px}.plan div{border:1px solid var(--line);border-radius:10px;padding:8px;background:var(--panel)}
.section{font-size:.68rem;font-weight:850;color:#b9c8d7;margin:12px 0 5px}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:8px}
.stButton>button{width:100%;border-radius:10px;background:#111b27;border:1px solid #26384c;color:white}
[data-testid="stTabs"] button{font-size:.72rem}
div[data-baseweb="select"]>div,.stTextInput input{background:#0d151f!important}
@media(max-width:700px){
.block-container{padding:.35rem .45rem 1.5rem!important}
.grid4{grid-template-columns:repeat(2,1fr)}.plan{grid-template-columns:repeat(2,1fr)}
.brand{font-size:1.05rem}.price{font-size:1.45rem}.score{font-size:1.65rem}
}
</style>
""", unsafe_allow_html=True)

def sf(v, default=np.nan):
    try:
        v=float(v); return v if np.isfinite(v) else default
    except: return default

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def rsi(s,n=14):
    d=s.diff()
    u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+u/dn.replace(0,np.nan))

def atr(d,n=14):
    pc=d.Close.shift()
    tr=pd.concat([(d.High-d.Low),(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def indicators(d):
    x=d.copy()
    x["EMA20"]=ema(x.Close,20); x["EMA60"]=ema(x.Close,60)
    x["RSI"]=rsi(x.Close); x["ATR"]=atr(x)
    m=ema(x.Close,12)-ema(x.Close,26); x["MACDH"]=m-ema(m,9)
    x["RVOL"]=x.Volume/x.Volume.rolling(20).mean().replace(0,np.nan)
    x["HH20"]=x.High.shift().rolling(20).max(); x["LL20"]=x.Low.shift().rolling(20).min()
    x["ROC10"]=x.Close.pct_change(10)*100
    return x

@st.cache_data(ttl=60,show_spinner=False)
def download_one(sym, period="30d", interval="5m"):
    try:
        d=yf.download(sym,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
        if d is None or d.empty:return None
        if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
        cols=["Open","High","Low","Close","Volume"]
        return d[cols].dropna() if all(c in d.columns for c in cols) else None
    except:return None

@st.cache_data(ttl=90,show_spinner=False)
def download_batch(symbols):
    try:
        raw=yf.download(" ".join(symbols),period="5d",interval="15m",auto_adjust=True,progress=False,threads=True,group_by="ticker")
        return raw
    except:return None

def resample(d,rule):
    return d.resample(rule).agg(Open=("Open","first"),High=("High","max"),Low=("Low","min"),Close=("Close","last"),Volume=("Volume","sum")).dropna()

def trend(d):
    if d is None or len(d)<60:return "N"
    x=indicators(d); r=x.iloc[-1]
    return "B" if r.Close>r.EMA20>r.EMA60 else ("S" if r.Close<r.EMA20<r.EMA60 else "N")

def signal(d):
    if d is None or len(d)<180:return None
    f=indicators(d); r=f.iloc[-1]; p=f.iloc[-2]
    f15=indicators(resample(d,"15min")); f60=indicators(resample(d,"60min"))
    if len(f60)<60:return None
    t5=trend(d); t15=trend(resample(d,"15min")); t60=trend(resample(d,"60min"))
    L=S=0; reasons=[]
    if t60=="B":L+=22
    elif t60=="S":S+=22
    if t15=="B":L+=15
    elif t15=="S":S+=15
    if t5=="B":L+=12
    elif t5=="S":S+=12
    if r.Close>r.EMA20:L+=8
    elif r.Close<r.EMA20:S+=8
    if r.MACDH>0 and r.MACDH>p.MACDH:L+=9
    elif r.MACDH<0 and r.MACDH<p.MACDH:S+=9
    if 52<=r.RSI<=72:L+=8
    elif 28<=r.RSI<=48:S+=8
    if pd.notna(r.RVOL) and r.RVOL>=1.2:
        if r.Close>r.Open:L+=9
        elif r.Close<r.Open:S+=9
    bull=pd.notna(r.HH20) and r.Close>r.HH20
    bear=pd.notna(r.LL20) and r.Close<r.LL20
    if bull:L+=10
    if bear:S+=10
    if pd.notna(r.ROC10):
        if r.ROC10>=1:L+=7
        elif r.ROC10<=-1:S+=7
    if t5==t15==t60=="B":L+=10
    if t5==t15==t60=="S":S+=10
    if bull and (pd.isna(r.RVOL) or r.RVOL<1.05):L-=6
    if bear and (pd.isna(r.RVOL) or r.RVOL<1.05):S-=6
    L=max(0,int(L));S=max(0,int(S));side="LONG" if L>=S else "SHORT"
    score=max(L,S);gap=abs(L-S)
    status="CONFIRMED" if score>=90 and gap>=20 else ("WAIT" if score>=82 and gap>=15 else ("WATCH" if score>=70 and gap>=10 else "NO TRADE"))
    entry=sf(r.Close); a=sf(r.ATR)
    if side=="LONG":
        stop=min(sf(f.Low.tail(8).min()),entry-1.35*a); risk=entry-stop
        targets=[entry+risk,entry+2*risk,entry+3*risk]
    else:
        stop=max(sf(f.High.tail(8).max()),entry+1.35*a); risk=stop-entry
        targets=[entry-risk,entry-2*risk,entry-3*risk]
    return dict(side=side,score=score,gap=gap,status=status,L=L,S=S,entry=entry,stop=stop,t1=targets[0],t2=targets[1],t3=targets[2],
                rsi=sf(r.RSI),rvol=sf(r.RVOL),roc=sf(r.ROC10),t5=t5,t15=t15,t60=t60,f=f.tail(120))

def tlabel(v):
    return {"B":"🟢 多","S":"🔴 空","N":"⚪ 中性"}.get(v,v)

def chart(s):
    f=s["f"]; fig=go.Figure()
    fig.add_trace(go.Candlestick(x=f.index,open=f.Open,high=f.High,low=f.Low,close=f.Close,name="Price"))
    fig.add_trace(go.Scatter(x=f.index,y=f.EMA20,name="EMA20",line=dict(width=1)))
    fig.add_trace(go.Scatter(x=f.index,y=f.EMA60,name="EMA60",line=dict(width=1)))
    for y,name in [(s["entry"],"ENTRY"),(s["stop"],"STOP"),(s["t1"],"TP1"),(s["t2"],"TP2"),(s["t3"],"TP3")]:
        fig.add_hline(y=y,line_width=1,line_dash="dot",annotation_text=name)
    fig.update_layout(height=380,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="#080d14",plot_bgcolor="#080d14",
                      font=dict(color="#9eb0c3",size=9),xaxis_rangeslider_visible=False,legend=dict(orientation="h"))
    return fig

st.markdown("""<div class="topbar"><div><div class="brand">⚡ QUANT <span>PRO 4.1</span></div>
<div class="sub">TECH 100 · MULTI-TIMEFRAME SIGNAL TERMINAL</div></div><div class="badge">LIVE ENGINE</div></div>""",unsafe_allow_html=True)

tabs=st.tabs(["⚡ 首页","🔥 TECH 100","🌐 市场","🛡️ 风控"])

with tabs[0]:
    c1,c2=st.columns([1,1])
    with c1:
        sym=st.selectbox("股票",TECH100,index=0)
    with c2:
        custom=st.text_input("自定义代码",placeholder="例如 CRWV")
        if custom.strip():sym=custom.strip().upper()
    d=download_one(sym)
    s=signal(d)
    if s is None:
        st.warning("当前数据不足或 Yahoo 暂时未返回足够的 5 分钟行情。")
    else:
        sidecls="green" if s["side"]=="LONG" else "red"
        st.markdown(f"""<div class="hero"><div class="hero-row"><div><div class="ticker">{sym}</div>
        <div class="price">${s['entry']:.2f}</div><div class="muted">5分钟量化执行价格</div></div>
        <div><div class="signal {sidecls}">{s['side']} · {s['status']}</div><div class="score">{s['score']}</div>
        <div class="muted" style="text-align:right">LONG {s['L']} · SHORT {s['S']} · GAP {s['gap']}</div></div></div>
        <div class="tf"><div>{tlabel(s['t5'])}<br><span class="muted">5 MIN</span></div>
        <div>{tlabel(s['t15'])}<br><span class="muted">15 MIN</span></div>
        <div>{tlabel(s['t60'])}<br><span class="muted">60 MIN</span></div>
        <div>{"🟢 强" if s["score"]>=90 else ("🟡 等待" if s["score"]>=82 else "⚪ 观察")}<br><span class="muted">SIGNAL</span></div></div></div>""",unsafe_allow_html=True)

        st.markdown('<div class="section">交易计划</div>',unsafe_allow_html=True)
        st.markdown(f"""<div class="plan"><div><div class="k">ENTRY</div><div class="v">${s['entry']:.2f}</div></div>
        <div><div class="k">STOP</div><div class="v red">${s['stop']:.2f}</div></div>
        <div><div class="k">TP1 · 1R</div><div class="v">${s['t1']:.2f}</div></div>
        <div><div class="k">TP2 · 2R</div><div class="v">${s['t2']:.2f}</div></div>
        <div><div class="k">TP3 · 3R</div><div class="v">${s['t3']:.2f}</div></div></div>""",unsafe_allow_html=True)

        st.markdown('<div class="section">技术状态</div>',unsafe_allow_html=True)
        st.markdown(f"""<div class="grid4"><div class="card"><div class="k">RSI</div><div class="v">{s['rsi']:.1f}</div></div>
        <div class="card"><div class="k">RVOL</div><div class="v">{s['rvol']:.2f}</div></div>
        <div class="card"><div class="k">ROC 10</div><div class="v">{s['roc']:.2f}%</div></div>
        <div class="card"><div class="k">方向差</div><div class="v">{s['gap']}</div></div></div>""",unsafe_allow_html=True)
        st.plotly_chart(chart(s),use_container_width=True,config={"displayModeBar":False})

with tabs[1]:
    st.markdown("### 🔥 TECH 100 信号雷达")
    st.caption("扫描固定 100 只热门科技/AI/芯片/软件/网络安全/机器人/量子/数据中心股票。深度 5m 多周期计算较重，默认分批扫描。")
    scan_n=st.select_slider("本次扫描数量",options=[10,20,30,50,100],value=20)
    mode=st.selectbox("显示",["全部有效信号","只看强确认 ≥90","只看 LONG","只看 SHORT"])
    if st.button(f"开始扫描 {scan_n} 只"):
        rows=[]; bar=st.progress(0); status=st.empty()
        for i,symx in enumerate(TECH100[:scan_n]):
            status.caption(f"正在分析 {i+1}/{scan_n} · {symx}")
            sx=signal(download_one(symx))
            if sx:
                ok=sx["score"]>=70
                if mode=="只看强确认 ≥90":ok=sx["score"]>=90 and sx["status"]=="CONFIRMED"
                elif mode=="只看 LONG":ok=ok and sx["side"]=="LONG"
                elif mode=="只看 SHORT":ok=ok and sx["side"]=="SHORT"
                if ok:
                    rows.append({"股票":symx,"方向":sx["side"],"评分":sx["score"],"状态":sx["status"],"Gap":sx["gap"],
                                 "价格":round(sx["entry"],2),"RVOL":round(sx["rvol"],2) if np.isfinite(sx["rvol"]) else None,
                                 "RSI":round(sx["rsi"],1)})
            bar.progress((i+1)/scan_n)
        status.empty();bar.empty()
        if rows:
            out=pd.DataFrame(rows).sort_values(["评分","Gap"],ascending=False).reset_index(drop=True)
            out.index=out.index+1
            st.markdown("#### 🏆 TOP 信号")
            st.dataframe(out,use_container_width=True,height=min(650,80+35*len(out)))
        else:st.info("本次扫描没有 ≥70 的有效信号。")

with tabs[2]:
    st.markdown("### 🌐 市场状态")
    cols=st.columns(3)
    for col,idx in zip(cols,["SPY","QQQ","IWM"]):
        md=download_one(idx); ms=signal(md)
        if ms: col.metric(idx,f"{ms['side']} {ms['score']}",f"Gap {ms['gap']}")
        else: col.metric(idx,"数据不足")
    st.caption("建议：大盘与个股方向一致时优先；V4.1 的 TECH 100 排名是候选信号，不等于自动下单指令。")

with tabs[3]:
    st.markdown("### 🛡️ 风险控制")
    account=st.number_input("账户资金 $",min_value=1000.0,value=20000.0,step=1000.0)
    risk=st.slider("单笔最大风险 %",0.1,2.0,0.5,0.1)
    daily=st.slider("日内最大亏损 %",0.5,5.0,2.0,0.5)
    c1,c2=st.columns(2)
    c1.metric("单笔风险上限",f"${account*risk/100:,.0f}")
    c2.metric("日内熔断",f"${account*daily/100:,.0f}")
    st.info("建议只把 ≥90 且 Gap ≥20 视为强确认候选。正式使用前应做历史回测、样本外测试，并考虑滑点、手续费、财报和宏观事件风险。")

st.caption(f"QUANT PRO 4.1 · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Research tool only · Not financial advice")
