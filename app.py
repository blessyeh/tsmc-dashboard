"""
台股 K線技術分析儀表板
支援日K / 週K / 月K 動態切換
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
    page_title="台股技術分析儀表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 各週期對應設定
# ─────────────────────────────────────────────
INTERVAL_CONFIG = {
    "日K": {
        "interval":   "1d",
        "period_map": {1:"1y", 2:"2y", 3:"3y", 4:"5y", 5:"5y"},
        "ma":         [20, 60, 120, 240],
        "ma_labels":  ["月線(20)", "季線(60)", "半年線(120)", "年線(240)"],
        "ma_colors":  ["#f5a623", "#7ed6df", "#e056fd", "#ff9f43"],
        "vol_ma":     5,
        "unit":       "日",
        "date_fmt":   "%Y/%m/%d",
        "bar_unit":   "今日",
        "prev_unit":  "昨日",
    },
    "週K": {
        "interval":   "1wk",
        "period_map": {1:"1y", 2:"2y", 3:"3y", 4:"5y", 5:"5y"},
        "ma":         [13, 26, 52],
        "ma_labels":  ["季線(13W)", "半年線(26W)", "年線(52W)"],
        "ma_colors":  ["#f5a623", "#7ed6df", "#e056fd"],
        "vol_ma":     5,
        "unit":       "週",
        "date_fmt":   "%Y/%m/%d",
        "bar_unit":   "本週",
        "prev_unit":  "上週",
    },
    "月K": {
        "interval":   "1mo",
        "period_map": {1:"2y", 2:"3y", 3:"5y", 4:"5y", 5:"max"},
        "ma":         [6, 12, 24, 60],
        "ma_labels":  ["半年(6M)", "年線(12M)", "2年(24M)", "5年(60M)"],
        "ma_colors":  ["#f5a623", "#7ed6df", "#e056fd", "#ff9f43"],
        "vol_ma":     3,
        "unit":       "月",
        "date_fmt":   "%Y/%m",
        "bar_unit":   "本月",
        "prev_unit":  "上月",
    },
}

# ─────────────────────────────────────────────
# 側邊欄設定
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 分析設定")
    ticker_input = st.text_input(
        "股票代碼",
        value="2330.TW",
        placeholder="例：2330.TW、TSM、AAPL",
        help="台股請加 .TW（如 2330.TW），上櫃加 .TWO，美股直接輸入代碼"
    )
    ticker = ticker_input.strip().upper() or "2330.TW"
    interval_label = st.radio("K線週期", ["日K", "週K", "月K"], index=1, horizontal=True)
    cfg = INTERVAL_CONFIG[interval_label]
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
def fetch_data(ticker, interval, period):
    t  = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    df = df[keep].copy()
    df.index = df.index.tz_localize(None)
    df.dropna(inplace=True)
    return df

# ─────────────────────────────────────────────
# 計算技術指標
# ─────────────────────────────────────────────
def calc_indicators(df, ma_periods, vol_ma_n):
    df = df.copy()
    for p in ma_periods:
        df[f'MA{p}'] = df['Close'].rolling(p).mean()

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

    df[f'Vol_MA{vol_ma_n}'] = df['Volume'].rolling(vol_ma_n).mean()
    return df

# ─────────────────────────────────────────────
# 偵測相對低點訊號
# ─────────────────────────────────────────────
def detect_signals(df, ma_periods, vol_ma_col, rsi_low=40, score_th=3):
    df   = df.copy()
    ma1  = f'MA{ma_periods[0]}'
    cond1 = df['RSI'] < rsi_low
    cond2 = df['Close'] <= df['BB_lower'] * 1.02
    cond3 = df['K'] < 25
    cond4 = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    cond5 = (df['MACD_hist'] > 0) & (df['MACD_hist'].shift(1) <= 0)
    cond6 = df['Close'] >= df[ma1] * 0.90
    cond7 = df['Volume'] < df[vol_ma_col]
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
def build_chart(df, cfg, interval_label, rsi_low):
    ma_periods = cfg['ma']
    vol_ma_col = f'Vol_MA{cfg["vol_ma"]}'

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.48, 0.18, 0.18, 0.16],
        subplot_titles=[
            f"{interval_label} + 均線 + 布林通道",
            "RSI(14)  |  KD(9,3,3)",
            "MACD(12,26,9)",
            "成交量"
        ]
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
        low=df['Low'], close=df['Close'], name=interval_label,
        increasing_line_color='#ff4444', decreasing_line_color='#22bb55',
        increasing_fillcolor='#ff4444', decreasing_fillcolor='#22bb55',
    ), row=1, col=1)

    # 均線
    for p, label, color in zip(ma_periods, cfg['ma_labels'], cfg['ma_colors']):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'MA{p}'], name=label,
            line=dict(color=color, width=1.5)), row=1, col=1)

    # 強訊號
    strong = df[df['signal_strong']]
    if not strong.empty:
        fig.add_trace(go.Scatter(
            x=strong.index, y=strong['Low'] * 0.975, mode='markers+text',
            name='低點訊號★',
            marker=dict(symbol='triangle-up', size=14, color='#00ff88',
                        line=dict(color='white', width=1)),
            text=['★'] * len(strong), textposition='bottom center',
            textfont=dict(size=10, color='#00ff88'),
            hovertemplate='<b>低點訊號</b><br>日期：%{x|%Y-%m-%d}<br>收盤：%{customdata[0]:.1f}<br>評分：%{customdata[1]}<extra></extra>',
            customdata=np.stack([strong['Close'], strong['signal_score']], axis=-1)
        ), row=1, col=1)

    # 中等訊號
    medium = df[df['signal_medium'] & ~df['signal_strong']]
    if not medium.empty:
        fig.add_trace(go.Scatter(
            x=medium.index, y=medium['Low'] * 0.975, mode='markers',
            name='觀察訊號△',
            marker=dict(symbol='triangle-up', size=9, color='#ffdd57',
                        line=dict(color='white', width=1))), row=1, col=1)

    # RSI + KD
    fig.add_hrect(y0=0, y1=rsi_low, row=2, col=1,
                  fillcolor="rgba(255,68,68,0.07)", line_width=0)
    for level, color, dash in [(rsi_low,'#ff6b6b','dash'),(50,'#888','dot'),(70,'#f5a623','dash')]:
        fig.add_hline(y=level, row=2, col=1, line=dict(color=color, width=1, dash=dash))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='#ff6b9d', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K',
        line=dict(color='#45aaf2', width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D',
        line=dict(color='#fd9644', width=1.5, dash='dot')), row=2, col=1)

    # MACD
    hist_colors = ['#ff4444' if v >= 0 else '#22bb55' for v in df['MACD_hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD柱',
        marker_color=hist_colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
        line=dict(color='#45aaf2', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='訊號線',
        line=dict(color='#fd9644', width=1.5, dash='dot')), row=3, col=1)
    fig.add_hline(y=0, row=3, col=1, line=dict(color='#555', width=1))

    # 成交量
    vol_c = ['#ff4444' if c >= o else '#22bb55'
              for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'] / 1e8, name='量(億股)',
        marker_color=vol_c, opacity=0.6), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[vol_ma_col] / 1e8,
        name=f'量MA{cfg["vol_ma"]}',
        line=dict(color='#f5a623', width=1.5)), row=4, col=1)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        height=860, hovermode='x unified',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02, x=0, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=20, t=60, b=40),
        font=dict(color='#c9d1d9', size=11)
    )
    fig.update_yaxes(gridcolor='#21262d')
    fig.update_xaxes(gridcolor='#21262d')
    return fig

# ─────────────────────────────────────────────
# 底部綜合說明
# ─────────────────────────────────────────────
def render_analysis(df, cfg, interval_label, rsi_low, score_th):
    last       = df.iloc[-1]
    prev       = df.iloc[-2]
    score      = int(last['signal_score'])
    bar_unit   = cfg['bar_unit']
    prev_unit  = cfg['prev_unit']
    ma_periods = cfg['ma']
    vol_ma_col = f'Vol_MA{cfg["vol_ma"]}'

    st.markdown("---")
    st.subheader("🔍 當前技術指標綜合分析說明")

    # 整體研判
    if score >= 4:
        icon, label = "🟢", "偏多（相對低點訊號明顯）"
        desc = f"目前共有 **{score} 項**技術指標同時出現低點共振，歷史上類似位置多為{interval_label}相對低點，可關注是否出現止跌訊號。"
    elif score == 3:
        icon, label = "🟡", "中性偏多（弱低點訊號）"
        desc = f"目前有 **{score} 項**指標出現低點訊號，訊號強度中等，建議搭配其他資訊綜合判斷。"
    elif score == 2:
        icon, label = "🟡", "中性（觀察訊號）"
        desc = f"目前有 **{score} 項**指標偏低點方向，尚未形成明顯共振，建議持續觀察後續走勢。"
    elif last['RSI'] > 65 or last['K'] > 75:
        icon, label = "🔴", "偏空（指標偏高）"
        desc = "目前多項指標處於相對高點，低點訊號評分偏低，不建議追高。"
    else:
        icon, label = "⚪", "中性（無明顯訊號）"
        desc = "目前各指標均處於中性區間，無明顯高低點訊號，適合觀望。"

    st.markdown(f"### {icon} 整體研判：{label}")
    st.info(desc)

    col_a, col_b = st.columns(2)

    with col_a:
        rsi_val = last['RSI']
        if rsi_val < rsi_low:
            rsi_txt = f"🟢 超賣區（{rsi_val:.1f}），動能偏弱但反彈機率上升"
        elif rsi_val > 70:
            rsi_txt = f"🔴 超買區（{rsi_val:.1f}），漲幅過大，留意回調風險"
        else:
            rsi_txt = f"🟡 中性區（{rsi_val:.1f}），無明顯超買超賣訊號"

        k_val, d_val = last['K'], last['D']
        kd_cross = ""
        if k_val > d_val and prev['K'] <= prev['D']:
            kd_cross = f"，**{bar_unit}出現黃金交叉**，動能轉正"
        elif k_val < d_val and prev['K'] >= prev['D']:
            kd_cross = f"，**{bar_unit}出現死亡交叉**，動能轉弱"
        kd_txt = (f"🟢 超賣區（K:{k_val:.1f} D:{d_val:.1f}）{kd_cross}" if k_val < 25 else
                  f"🔴 超買區（K:{k_val:.1f} D:{d_val:.1f}）{kd_cross}" if k_val > 75 else
                  f"🟡 中性（K:{k_val:.1f} D:{d_val:.1f}）{kd_cross}")

        h_now, h_prev = last['MACD_hist'], prev['MACD_hist']
        if h_now > 0 and h_prev <= 0:
            macd_txt = f"🟢 MACD柱{bar_unit}由負轉正（{h_now:.2f}），動能反轉訊號"
        elif h_now < 0 and h_prev >= 0:
            macd_txt = f"🔴 MACD柱{bar_unit}由正轉負（{h_now:.2f}），動能開始走弱"
        elif h_now > 0:
            macd_txt = f"🟢 MACD柱持續為正（{h_now:.2f}），多頭動能延續"
        else:
            macd_txt = f"🔴 MACD柱持續為負（{h_now:.2f}），空頭動能延續中"

        st.markdown("**📊 動能指標**")
        st.markdown(f"- **RSI(14)**：{rsi_txt}")
        st.markdown(f"- **KD(9,3,3)**：{kd_txt}")
        st.markdown(f"- **MACD**：{macd_txt}")

    with col_b:
        bb_pct = last['BB_pct'] * 100
        bb_txt = (f"🟢 接近下軌（%B:{bb_pct:.1f}%，下軌 {last['BB_lower']:.1f}），偏離均值大，回歸機率提升" if bb_pct < 20 else
                  f"🔴 接近上軌（%B:{bb_pct:.1f}%，上軌 {last['BB_upper']:.1f}），留意過熱風險" if bb_pct > 80 else
                  f"🟡 位於通道中段（%B:{bb_pct:.1f}%），無極端偏離")

        close_val  = last['Close']
        ma_vals    = [last[f'MA{p}'] for p in ma_periods]
        above      = sum(1 for m in ma_vals if close_val > m)
        ma_icon    = "🟢" if above == len(ma_periods) else ("🟡" if above >= 1 else "🔴")
        ma_detail  = "、".join(
            f"{'站上' if close_val > m else '跌破'}{cfg['ma_labels'][i].split('(')[0]}（{m:.1f}）"
            for i, (p, m) in enumerate(zip(ma_periods, ma_vals))
        )
        ma_txt = f"{ma_icon} 站上 {above}/{len(ma_periods)} 條均線：{ma_detail}"

        vol_ratio = last['Volume'] / last[vol_ma_col] if last[vol_ma_col] > 0 else 1
        vol_txt = (f"🟢 量能萎縮（均量的 {vol_ratio*100:.0f}%），籌碼沉澱，低點特徵之一" if vol_ratio < 0.7 else
                   f"🔴 量能放大（均量的 {vol_ratio*100:.0f}%），需觀察是否量價背離" if vol_ratio > 1.5 else
                   f"🟡 量能正常（均量的 {vol_ratio*100:.0f}%），無異常")

        st.markdown("**📐 結構指標**")
        st.markdown(f"- **布林通道**：{bb_txt}")
        st.markdown(f"- **均線位置**：{ma_txt}")
        st.markdown(f"- **成交量**：{vol_txt}")

    # 低點評分明細
    with st.expander("📋 低點評分明細（共 7 項條件）"):
        ma1_label  = cfg['ma_labels'][0].split('(')[0]
        ma1_col    = f'MA{ma_periods[0]}'
        vol_n      = cfg['vol_ma']
        cond_list  = [
            ("RSI < 超賣門檻",
             bool(last['RSI'] < rsi_low),
             f"RSI={last['RSI']:.1f}，門檻={rsi_low}"),
            ("收盤 ≤ 布林下軌 ×1.02",
             bool(last['Close'] <= last['BB_lower'] * 1.02),
             f"收盤={last['Close']:.1f}，下軌={last['BB_lower']:.1f}"),
            ("K值 < 25（KD超賣）",
             bool(last['K'] < 25),
             f"K={last['K']:.1f}"),
            ("KD 黃金交叉",
             bool(last['K'] > last['D'] and prev['K'] <= prev['D']),
             f"K={last['K']:.1f} D={last['D']:.1f}"),
            ("MACD柱由負轉正",
             bool(last['MACD_hist'] > 0 and prev['MACD_hist'] <= 0),
             f"{bar_unit}={last['MACD_hist']:.2f}，{prev_unit}={prev['MACD_hist']:.2f}"),
            (f"收盤 ≥ {ma1_label} ×0.9",
             bool(last['Close'] >= last[ma1_col] * 0.9),
             f"收盤={last['Close']:.1f}，{ma1_label}×0.9={last[ma1_col]*0.9:.1f}"),
            (f"成交量低於{vol_n}期均量",
             bool(last['Volume'] < last[vol_ma_col]),
             f"{bar_unit}量={last['Volume']/1e8:.2f}億，均量={last[vol_ma_col]/1e8:.2f}億"),
        ]
        for name, passed, detail in cond_list:
            st.markdown(f"{'✅' if passed else '❌'} **{name}**　*（{detail}）*")
        verdict = ("→ 強低點訊號" if score >= score_th else
                   "→ 觀察訊號" if score == score_th - 1 else "→ 無明顯低點訊號")
        st.markdown(f"**合計：{score} / 7 項條件成立　{verdict}**")

    st.markdown("> ⚠️ **免責聲明**：以上分析純為技術面參考，不構成任何投資建議。股市有風險，投資需謹慎。")


# ─────────────────────────────────────────────
# 主介面
# ─────────────────────────────────────────────
period     = cfg['period_map'].get(years, '3y')
vol_ma_col = f'Vol_MA{cfg["vol_ma"]}'

st.title(f"📈 {ticker}　{interval_label} 技術分析儀表板")

with st.spinner("📡 正在抓取最新股價資料..."):
    try:
        df = fetch_data(ticker, cfg['interval'], period)
        if df.empty:
            st.error("無法取得資料，請稍後再試")
            st.stop()
        df = calc_indicators(df, cfg['ma'], cfg['vol_ma'])
        df = detect_signals(df, cfg['ma'], vol_ma_col, rsi_low, score_th)
    except Exception as e:
        st.error(f"資料抓取失敗：{e}")
        st.stop()

last = df.iloc[-1]

# 指標卡片
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("收盤價", f"{last['Close']:.1f}",
              f"{last['Close'] - df.iloc[-2]['Close']:+.1f}")
with c2:
    st.metric("RSI(14)", f"{last['RSI']:.1f}",
              "超賣↑" if last['RSI'] < rsi_low else ("超買↓" if last['RSI'] > 70 else "中性"))
with c3:
    st.metric("K值", f"{last['K']:.1f}",
              "超賣" if last['K'] < 25 else ("超買" if last['K'] > 75 else ""))
with c4:
    st.metric("BB %B", f"{last['BB_pct']*100:.1f}%",
              "接近下軌" if last['BB_pct'] < 0.2 else ("接近上軌" if last['BB_pct'] > 0.8 else ""))
with c5:
    score_now = int(last['signal_score'])
    st.metric("低點評分", f"{score_now} / 7",
              "★ 強訊號" if last['signal_strong'] else ("△ 觀察" if last['signal_medium'] else ""))

# 圖表
fig = build_chart(df, cfg, interval_label, rsi_low)
st.plotly_chart(fig, use_container_width=True)

# 近期低點訊號表
strong_signals = df[df['signal_strong']].tail(5)
if not strong_signals.empty:
    st.subheader(f"📍 近期強低點訊號（{interval_label}，供參考）")
    disp = strong_signals[['Close','RSI','K','D','BB_pct','signal_score']].copy()
    disp.index = disp.index.strftime(cfg['date_fmt'])
    disp.columns = ['收盤價','RSI','K值','D值','BB%B','評分']
    disp['BB%B'] = (disp['BB%B'] * 100).round(1)
    st.dataframe(disp.style.format({
        '收盤價':'{:.1f}', 'RSI':'{:.1f}', 'K值':'{:.1f}',
        'D值':'{:.1f}', 'BB%B':'{:.1f}%', '評分':'{:.0f}'
    }), use_container_width=True)

st.caption(f"⏱ 資料更新：{df.index[-1].strftime(cfg['date_fmt'])}  |  週期：{interval_label}  |  ⚠️ 僅供技術分析參考")

# 底部綜合說明
render_analysis(df, cfg, interval_label, rsi_low, score_th)
