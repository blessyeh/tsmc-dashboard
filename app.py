"""
台積電 (2330.TW) 週K 完整技術分析儀表板
Streamlit 版本 - 可部署至 Streamlit Cloud（免費）
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="台積電週K技術分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 側邊欄設定
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 分析設定")
    ticker   = st.selectbox("股票代碼", ["2330.TW", "2317.TW", "2454.TW", "TSM"], index=0)
    years    = st.slider("歷史資料年數", 1, 5, 3)
    rsi_low  = st.slider("RSI 超賣門檻", 20, 45, 40)
    score_th = st.slider("低點訊號最低評分", 2, 5, 3)
    st.markdown("---")
    st.caption("資料來源：Yahoo Finance")
    st.caption("僅供參考，非投資建議")

# ─────────────────────────────────────────────
# 資料抓取（快取 30 分鐘）
# ─────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_data(ticker, years):
    # 用 Ticker.history() 取代 yf.download()
    # 對台股（.TW / .TWO）相容性更好，不會有 MultiIndex 欄位問題
    t   = yf.Ticker(ticker)
    period_map = {1: "1y", 2: "2y", 3: "3y", 4: "5y", 5: "5y"}
    period = period_map.get(years, "3y")
    df = t.history(period=period, interval="1wk", auto_adjust=True)

    # history() 回傳欄位名稱固定為英文，直接使用
    # 保留需要的欄位，移除 Dividends / Stock Splits
    keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    df = df[keep].copy()
    df.index = df.index.tz_localize(None)   # 去除時區資訊，避免後續比較問題
    df.dropna(inplace=True)
    return df

# ─────────────────────────────────────────────
# 計算技術指標
# ─────────────────────────────────────────────
def calc_indicators(df):
    df = df.copy()
    df['MA13']  = df['Close'].rolling(13).mean()
    df['MA26']  = df['Close'].rolling(26).mean()
    df['MA52']  = df['Close'].rolling(52).mean()

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    df['BB_mid']   = df['Close'].rolling(20).mean()
    std            = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * std
    df['BB_lower'] = df['BB_mid'] - 2 * std
    df['BB_pct']   = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    low9  = df['Low'].rolling(9).min()
    high9 = df['High'].rolling(9).max()
    rsv   = (df['Close'] - low9) / (high9 - low9 + 1e-9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    df['Vol_MA5'] = df['Volume'].rolling(5).mean()
    return df

# ─────────────────────────────────────────────
# 偵測相對低點訊號
# ─────────────────────────────────────────────
def detect_signals(df, rsi_low=40, score_th=3):
    df = df.copy()
    cond1 = df['RSI'] < rsi_low
    cond2 = df['Close'] <= df['BB_lower'] * 1.02
    cond3 = df['K'] < 25
    cond4 = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    cond5 = (df['MACD_hist'] > 0) & (df['MACD_hist'].shift(1) <= 0)
    cond6 = df['Close'] >= df['MA13'] * 0.90
    cond7 = df['Volume'] < df['Vol_MA5']

    score = (cond1.astype(int) + cond2.astype(int) + cond3.astype(int) +
             cond4.astype(int) + cond5.astype(int) + cond6.astype(int) +
             cond7.astype(int))

    df['signal_strong'] = score >= score_th
    df['signal_medium'] = score == (score_th - 1)
    df['signal_score']  = score
    return df

# ─────────────────────────────────────────────
# 建立 Plotly 圖表
# ─────────────────────────────────────────────
def build_chart(df):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.48, 0.18, 0.18, 0.16],
        subplot_titles=["週K + 均線 + 布林通道", "RSI(14)  |  KD(9,3,3)", "MACD(12,26,9)", "成交量"]
    )

    # 布林通道
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'],
        line=dict(color='rgba(200,200,200,0.25)', width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'],
        fill='tonexty', fillcolor='rgba(173,216,230,0.10)',
        line=dict(color='rgba(200,200,200,0.25)', width=1), name='布林通道'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_mid'],
        line=dict(color='#8888aa', width=1, dash='dot'), showlegend=False), row=1, col=1)

    # K線
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='週K',
        increasing_line_color='#ff4444', decreasing_line_color='#22bb55',
        increasing_fillcolor='#ff4444', decreasing_fillcolor='#22bb55',
    ), row=1, col=1)

    # 均線
    for ma, color, name in [('MA13','#f5a623','季線13W'),('MA26','#7ed6df','半年線26W'),('MA52','#e056fd','年線52W')]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=name,
            line=dict(color=color, width=1.5)), row=1, col=1)

    # 強訊號
    strong = df[df['signal_strong']]
    if not strong.empty:
        fig.add_trace(go.Scatter(
            x=strong.index, y=strong['Low'] * 0.975, mode='markers+text',
            name='低點訊號★', marker=dict(symbol='triangle-up', size=14, color='#00ff88',
            line=dict(color='white', width=1)),
            text=['★']*len(strong), textposition='bottom center',
            textfont=dict(size=10, color='#00ff88'),
            hovertemplate='<b>低點訊號</b><br>日期：%{x|%Y-%m-%d}<br>收盤：%{customdata[0]:.1f}<br>評分：%{customdata[1]}<extra></extra>',
            customdata=np.stack([strong['Close'], strong['signal_score']], axis=-1)
        ), row=1, col=1)

    # 中等訊號
    medium = df[df['signal_medium'] & ~df['signal_strong']]
    if not medium.empty:
        fig.add_trace(go.Scatter(
            x=medium.index, y=medium['Low'] * 0.975, mode='markers',
            name='觀察訊號△', marker=dict(symbol='triangle-up', size=9, color='#ffdd57',
            line=dict(color='white', width=1))), row=1, col=1)

    # RSI
    fig.add_hrect(y0=0, y1=rsi_low, row=2, col=1, fillcolor="rgba(255,68,68,0.07)", line_width=0)
    for level, color, dash in [(rsi_low,'#ff6b6b','dash'),(50,'#888','dot'),(70,'#f5a623','dash')]:
        fig.add_hline(y=level, row=2, col=1, line=dict(color=color, width=1, dash=dash))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#ff6b9d', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['K'],   name='K',   line=dict(color='#45aaf2', width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'],   name='D',   line=dict(color='#fd9644', width=1.5, dash='dot')), row=2, col=1)

    # MACD
    hist_colors = ['#ff4444' if v >= 0 else '#22bb55' for v in df['MACD_hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD柱',
        marker_color=hist_colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],        name='MACD',  line=dict(color='#45aaf2', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='訊號線', line=dict(color='#fd9644', width=1.5, dash='dot')), row=3, col=1)
    fig.add_hline(y=0, row=3, col=1, line=dict(color='#555', width=1))

    # 成交量
    vol_c = ['#ff4444' if c >= o else '#22bb55' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume']/1e8, name='量(億股)',
        marker_color=vol_c, opacity=0.6), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Vol_MA5']/1e8, name='量MA5',
        line=dict(color='#f5a623', width=1.5)), row=4, col=1)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        height=850, hovermode='x unified',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02, x=0, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=60, b=40),
        font=dict(color='#c9d1d9', size=11)
    )
    fig.update_yaxes(gridcolor='#21262d')
    fig.update_xaxes(gridcolor='#21262d')
    return fig

# ─────────────────────────────────────────────
# 主介面
# ─────────────────────────────────────────────
st.title(f"📈 {ticker} 週K 技術分析儀表板")

with st.spinner("📡 正在抓取最新股價資料..."):
    try:
        df = fetch_data(ticker, years)
        if df.empty:
            st.error("無法取得資料，請稍後再試")
            st.stop()
        df = calc_indicators(df)
        df = detect_signals(df, rsi_low, score_th)
    except Exception as e:
        st.error(f"資料抓取失敗：{e}")
        st.stop()

last = df.iloc[-1]

# ── 最新指標卡片 ──────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

def delta_color(val, good_low, good_high):
    if val <= good_low:  return "normal"   # 綠色（超賣）
    if val >= good_high: return "inverse"  # 紅色（超買）
    return "off"

with col1:
    st.metric("收盤價", f"{last['Close']:.1f}", 
              f"{last['Close']-df.iloc[-2]['Close']:+.1f}")
with col2:
    st.metric("RSI(14)", f"{last['RSI']:.1f}",
              "超賣 ↑" if last['RSI'] < rsi_low else ("超買 ↓" if last['RSI'] > 70 else "中性"))
with col3:
    st.metric("K值", f"{last['K']:.1f}",
              "超賣" if last['K'] < 25 else ("超買" if last['K'] > 75 else ""))
with col4:
    st.metric("BB %B", f"{last['BB_pct']*100:.1f}%",
              "接近下軌" if last['BB_pct'] < 0.2 else ("接近上軌" if last['BB_pct'] > 0.8 else ""))
with col5:
    score = int(last['signal_score'])
    st.metric("低點評分", f"{score} / 7",
              "★ 強訊號" if last['signal_strong'] else ("△ 觀察" if last['signal_medium'] else ""))

# ── 圖表 ──────────────────────────────────────────
fig = build_chart(df)
st.plotly_chart(fig, use_container_width=True)

# ── 近期低點訊號表 ────────────────────────────────
strong_signals = df[df['signal_strong']].tail(5)
if not strong_signals.empty:
    st.subheader("📍 近期強低點訊號（供參考）")
    display_df = strong_signals[['Close','RSI','K','D','BB_pct','signal_score']].copy()
    display_df.index = display_df.index.strftime('%Y/%m/%d')
    display_df.columns = ['收盤價','RSI','K值','D值','BB%B','評分']
    display_df['BB%B'] = (display_df['BB%B'] * 100).round(1)
    st.dataframe(display_df.style.format({
        '收盤價':'{:.1f}', 'RSI':'{:.1f}', 'K值':'{:.1f}',
        'D值':'{:.1f}', 'BB%B':'{:.1f}%', '評分':'{:.0f}'
    }), use_container_width=True)

st.caption(f"⏱ 資料更新：{df.index[-1].strftime('%Y/%m/%d')}  |  ⚠️ 本工具僅供技術分析參考，非投資建議")
