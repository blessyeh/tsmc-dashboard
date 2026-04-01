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
    ticker   = st.selectbox("股票代碼", ["2330.TW", "2850.TW","2317.TW", "2454.TW", "TSM"], index=0)
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

# ─────────────────────────────────────────────
# 當前指標綜合說明
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 當前技術指標綜合分析說明")

last  = df.iloc[-1]
prev  = df.iloc[-2]
score = int(last['signal_score'])

# ── 整體研判 ──────────────────────────────────
if score >= 4:
    overall_icon  = "🟢"
    overall_label = "偏多（相對低點訊號明顯）"
    overall_desc  = f"目前共有 **{score} 項**技術指標同時出現低點共振，歷史上類似位置多為中線相對低點，可關注是否出現止跌訊號。"
elif score == 3:
    overall_icon  = "🟡"
    overall_label = "中性偏多（弱低點訊號）"
    overall_desc  = f"目前有 **{score} 項**指標出現低點訊號，訊號強度中等，建議搭配其他資訊綜合判斷，尚未到強力介入時機。"
elif score == 2:
    overall_icon  = "🟡"
    overall_label = "中性（觀察訊號）"
    overall_desc  = f"目前有 **{score} 項**指標偏低點方向，尚未形成明顯共振，建議持續觀察下週走勢。"
elif last['RSI'] > 65 or last['K'] > 75:
    overall_icon  = "🔴"
    overall_label = "偏空（指標偏高）"
    overall_desc  = "目前多項指標處於相對高點或中性區間，低點訊號評分偏低，不建議追高。"
else:
    overall_icon  = "⚪"
    overall_label = "中性（無明顯訊號）"
    overall_desc  = "目前各指標均處於中性區間，無明顯高低點訊號，適合觀望等待更清晰的方向。"

st.markdown(f"### {overall_icon} 整體研判：{overall_label}")
st.info(overall_desc)

# ── 各指標分析 ────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    rsi_val = last['RSI']
    if rsi_val < rsi_low:
        rsi_status = f"🟢 超賣區（{rsi_val:.1f}），動能偏弱但反彈機率上升"
    elif rsi_val > 70:
        rsi_status = f"🔴 超買區（{rsi_val:.1f}），短期漲幅過大，留意回調風險"
    else:
        rsi_status = f"🟡 中性區（{rsi_val:.1f}），無明顯超買超賣訊號"

    k_val, d_val = last['K'], last['D']
    kd_cross = ""
    if k_val > d_val and prev['K'] <= prev['D']:
        kd_cross = "，**本週出現黃金交叉**，動能轉正"
    elif k_val < d_val and prev['K'] >= prev['D']:
        kd_cross = "，**本週出現死亡交叉**，動能轉弱"
    if k_val < 25:
        kd_status = f"🟢 超賣區（K:{k_val:.1f} D:{d_val:.1f}）{kd_cross}"
    elif k_val > 75:
        kd_status = f"🔴 超買區（K:{k_val:.1f} D:{d_val:.1f}）{kd_cross}"
    else:
        kd_status = f"🟡 中性（K:{k_val:.1f} D:{d_val:.1f}）{kd_cross}"

    hist_val  = last['MACD_hist']
    hist_prev = prev['MACD_hist']
    if hist_val > 0 and hist_prev <= 0:
        macd_status = f"🟢 MACD柱本週由負轉正（{hist_val:.2f}），動能出現反轉訊號"
    elif hist_val < 0 and hist_prev >= 0:
        macd_status = f"🔴 MACD柱本週由正轉負（{hist_val:.2f}），動能開始走弱"
    elif hist_val > 0:
        macd_status = f"🟢 MACD柱持續為正（{hist_val:.2f}），多頭動能延續"
    else:
        macd_status = f"🔴 MACD柱持續為負（{hist_val:.2f}），空頭動能延續中"

    st.markdown("**📊 動能指標**")
    st.markdown(f"- **RSI(14)**：{rsi_status}")
    st.markdown(f"- **KD(9,3,3)**：{kd_status}")
    st.markdown(f"- **MACD**：{macd_status}")

with col_b:
    bb_pct  = last['BB_pct'] * 100
    bb_low  = last['BB_lower']
    bb_high = last['BB_upper']
    if bb_pct < 20:
        bb_status = f"🟢 接近下軌（%B:{bb_pct:.1f}%，下軌 {bb_low:.1f}），統計上偏離均值過大，回歸機率提升"
    elif bb_pct > 80:
        bb_status = f"🔴 接近上軌（%B:{bb_pct:.1f}%，上軌 {bb_high:.1f}），留意過熱風險"
    else:
        bb_status = f"🟡 位於通道中段（%B:{bb_pct:.1f}%），無極端偏離"

    close_val = last['Close']
    ma13, ma26, ma52 = last['MA13'], last['MA26'], last['MA52']
    above_count = sum(1 for m in [ma13, ma26, ma52] if close_val > m)
    ma_icon = "🟢" if above_count == 3 else ("🟡" if above_count >= 1 else "🔴")
    ma_lines = []
    ma_lines.append(f"{'站上' if close_val > ma13 else '跌破'}季線（{ma13:.1f}）")
    ma_lines.append(f"{'站上' if close_val > ma26 else '跌破'}半年線（{ma26:.1f}）")
    ma_lines.append(f"{'站上' if close_val > ma52 else '跌破'}年線（{ma52:.1f}）")
    ma_status = f"{ma_icon} 站上 {above_count}/3 條均線：" + "、".join(ma_lines)

    vol_val   = last['Volume']
    vol_ma5   = last['Vol_MA5']
    vol_ratio = vol_val / vol_ma5 if vol_ma5 > 0 else 1
    if vol_ratio < 0.7:
        vol_status = f"🟢 量能萎縮（均量的 {vol_ratio*100:.0f}%），籌碼沉澱，低點特徵之一"
    elif vol_ratio > 1.5:
        vol_status = f"🔴 量能放大（均量的 {vol_ratio*100:.0f}%），需觀察是否量價背離"
    else:
        vol_status = f"🟡 量能正常（均量的 {vol_ratio*100:.0f}%），無異常"

    st.markdown("**📐 結構指標**")
    st.markdown(f"- **布林通道**：{bb_status}")
    st.markdown(f"- **均線位置**：{ma_status}")
    st.markdown(f"- **成交量**：{vol_status}")

# ── 低點評分明細 ──────────────────────────────
with st.expander("📋 低點評分明細（共 7 項條件）"):
    cond_results = [
        ("RSI < 超賣門檻",        bool(last['RSI'] < rsi_low),                           f"RSI={last['RSI']:.1f}，門檻={rsi_low}"),
        ("收盤 ≤ 布林下軌 ×1.02", bool(last['Close'] <= last['BB_lower']*1.02),           f"收盤={last['Close']:.1f}，下軌={last['BB_lower']:.1f}"),
        ("K值 < 25（KD超賣）",     bool(last['K'] < 25),                                  f"K={last['K']:.1f}"),
        ("KD 黃金交叉",            bool(last['K'] > last['D'] and prev['K'] <= prev['D']),f"K={last['K']:.1f} D={last['D']:.1f}"),
        ("MACD柱由負轉正",         bool(last['MACD_hist'] > 0 and prev['MACD_hist'] <= 0),f"本週={last['MACD_hist']:.2f}，上週={prev['MACD_hist']:.2f}"),
        ("收盤 ≥ 季線 ×0.9",       bool(last['Close'] >= last['MA13']*0.9),               f"收盤={last['Close']:.1f}，季線90%={last['MA13']*0.9:.1f}"),
        ("成交量低於5週均量",       bool(last['Volume'] < last['Vol_MA5']),                f"本週量={last['Volume']/1e8:.2f}億，均量={last['Vol_MA5']/1e8:.2f}億"),
    ]
    for name, passed, detail in cond_results:
        icon = "✅" if passed else "❌"
        st.markdown(f"{icon} **{name}**　*（{detail}）*")
    label = "→ 強低點訊號" if score >= score_th else ("→ 觀察訊號" if score == score_th-1 else "→ 無明顯低點訊號")
    st.markdown(f"**合計：{score} / 7 項條件成立　{label}**")

st.markdown("> ⚠️ **免責聲明**：以上分析純為技術面參考，不構成任何投資建議。股市有風險，投資需謹慎。")
