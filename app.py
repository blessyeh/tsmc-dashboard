"""
台股 K線技術分析儀表板 v2（升級版）
新增：趨勢濾網、結構支撐、量能雙條件、進場觸發、市場環境、外資籌碼
評分制度：10分制（趨勢2+動能3+結構3+量能1+觸發1）
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="台股技術分析儀表板 v2",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 各週期設定
# ─────────────────────────────────────────────
INTERVAL_CONFIG = {
    "日K": {
        "interval":      "1d",
        "period_map":    {1:"1y", 2:"2y", 3:"3y", 4:"5y", 5:"5y"},
        "ma":            [20, 60, 120, 240],
        "ma_labels":     ["月線(20)", "季線(60)", "半年線(120)", "年線(240)"],
        "ma_colors":     ["#f5a623", "#7ed6df", "#e056fd", "#ff9f43"],
        "vol_ma":        5,
        "trigger_ma":    5,
        "slope_periods": 4,
        "support_periods": 20,
        "date_fmt":      "%Y/%m/%d",
        "bar_unit":      "今日",
        "prev_unit":     "昨日",
    },
    "週K": {
        "interval":      "1wk",
        "period_map":    {1:"1y", 2:"2y", 3:"3y", 4:"5y", 5:"5y"},
        "ma":            [13, 26, 52],
        "ma_labels":     ["季線(13W)", "半年線(26W)", "年線(52W)"],
        "ma_colors":     ["#f5a623", "#7ed6df", "#e056fd"],
        "vol_ma":        5,
        "trigger_ma":    5,
        "slope_periods": 4,
        "support_periods": 20,
        "date_fmt":      "%Y/%m/%d",
        "bar_unit":      "本週",
        "prev_unit":     "上週",
    },
    "月K": {
        "interval":      "1mo",
        "period_map":    {1:"2y", 2:"3y", 3:"5y", 4:"5y", 5:"max"},
        "ma":            [6, 12, 24, 60],
        "ma_labels":     ["半年(6M)", "年線(12M)", "2年(24M)", "5年(60M)"],
        "ma_colors":     ["#f5a623", "#7ed6df", "#e056fd", "#ff9f43"],
        "vol_ma":        3,
        "trigger_ma":    3,
        "slope_periods": 3,
        "support_periods": 12,
        "date_fmt":      "%Y/%m",
        "bar_unit":      "本月",
        "prev_unit":     "上月",
    },
}

# ─────────────────────────────────────────────
# 側邊欄
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("?? 分析設定")

    ticker_input = st.text_input(
        "股票代碼",
        value="2330.TW",
        placeholder="例：2330.TW、TSM、AAPL",
        help="台股加 .TW（如 2330.TW），上櫃加 .TWO，美股直接輸入代碼"
    )
    ticker = ticker_input.strip().upper() or "2330.TW"

    interval_label = st.radio("K線週期", ["日K", "週K", "月K"], index=1, horizontal=True)
    cfg = INTERVAL_CONFIG[interval_label]

    years    = st.slider("歷史資料年數", 1, 5, 3)
    rsi_low  = st.slider("RSI 超賣門檻", 20, 45, 40)
    score_th = st.slider("強訊號最低評分（/10）", 3, 8, 6)
    use_trend_filter = st.checkbox("啟用趨勢濾網（推薦）", value=True,
                                   help="空頭市場時，強訊號標記會改為灰色警示，避免逆勢操作")

    st.markdown("---")
    st.caption("資料來源：Yahoo Finance / TWSE")
    st.caption("僅供參考，非投資建議")

# ─────────────────────────────────────────────
# 資料抓取
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

@st.cache_data(ttl=1800)
def fetch_market_env(interval):
    """用 0050.TW 作為大盤環境代理"""
    try:
        t  = yf.Ticker("0050.TW")
        df = t.history(period="2y", interval=interval, auto_adjust=True)
        keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
        df = df[keep].copy()
        df.index = df.index.tz_localize(None)
        df.dropna(inplace=True)
        if len(df) < 20:
            return None

        annual_n = min(52, len(df) // 2)
        df['MA_annual'] = df['Close'].rolling(annual_n).mean()

        delta = df['Close'].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        last = df.iloc[-1]
        above = bool(last['Close'] > last['MA_annual'])
        rsi_v = float(last['RSI'])
        return {
            'close':        float(last['Close']),
            'above_annual': above,
            'rsi':          rsi_v,
            'bullish':      above and rsi_v > 50,
        }
    except:
        return None

def _parse_twse_net(raw: str) -> int:
    """將 TWSE 回傳數字字串轉整數（處理全形負號、千分位逗號）"""
    s = raw.replace(',', '').replace('?', '-').replace('－', '-').strip()
    try:
        return int(s) if s and s not in ('--', '') else 0
    except ValueError:
        return 0

def _recent_trading_days(n: int) -> list:
    """回傳最近 n 個交易日清單（跳過週六日）格式 YYYYMMDD"""
    dates, d = [], datetime.now()
    while len(dates) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d.strftime('%Y%m%d'))
    return dates

@st.cache_data(ttl=1800)
def fetch_institutional(ticker):
    """
    外資買賣超：FinMind API 為主，TWSE OpenAPI 為備援
    回傳 dict 或 {'error': str} 以便前端顯示診斷訊息
    """
    if not ticker.endswith('.TW'):
        return None
    stock_code = ticker.replace('.TW', '')
    start_date = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')

    # ── 方法 A：FinMind 公開 API ──────────────────────────────────────────────
    # FinMind name 欄位實際值（依版本不同可能為以下任一）：
    #   "外資及陸資(不含外資自營商)"、"外陸資(不含外資自營商)"、"外資自營商"
    # 用最寬鬆的 '外資' 關鍵字比對，避免漏抓
    try:
        url = (
            'https://api.finmindtrade.com/api/v4/data'
            f'?dataset=TaiwanStockInstitutionalInvestorsBuySell'
            f'&data_id={stock_code}'
            f'&start_date={start_date}'
        )
        resp = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })
        if resp.status_code == 200:
            payload = resp.json()
            raw_data = payload.get('data', [])
            if payload.get('status') == 200 and raw_data:
                from collections import defaultdict
                daily      = defaultdict(int)
                seen_names = set()
                for row in raw_data:
                    name = row.get('name', '')
                    seen_names.add(name)
                    # 比對外資（含所有外資相關法人，排除投信/自營商）
                    if name in ('Foreign_Investor', 'Foreign_Dealer_Self'):
                        net = (int(row.get('buy', 0)) - int(row.get('sell', 0))) // 1000
                        daily[row['date']] += net
                if daily:
                    sorted_dates = sorted(daily.keys(), reverse=True)[:10]
                    records = [{'date': d, 'foreign_net': daily[d]} for d in sorted_dates]
                    for r in records:
                        r['is_buy'] = r['foreign_net'] > 0
                    consecutive_buy = 0
                    for r in records:
                        if r['is_buy']: consecutive_buy += 1
                        else: break
                    total_5d = sum(r['foreign_net'] for r in records[:5])
                    return {
                        'source':          'FinMind',
                        'records':         records[:10],
                        'consecutive_buy': consecutive_buy,
                        'total_net_5d':    total_5d,
                        'latest_net':      records[0]['foreign_net'],
                        'bullish':         consecutive_buy >= 3 or (consecutive_buy >= 1 and total_5d > 0),
                    }
                # FinMind 回傳資料但比對不到外資，附上實際名稱供診斷
                finmind_err = f"FinMind 有資料但外資名稱未匹配，實際 name 值：{seen_names}"
            else:
                finmind_err = f"FinMind status={payload.get('status')}, msg={payload.get('msg','')}, rows={len(raw_data)}"
        else:
            finmind_err = f"FinMind HTTP {resp.status_code}"
    except Exception as e:
        finmind_err = f"FinMind exception: {e}"

    twse_err = "TWSE OpenAPI 已停用（回傳空值）"

    # 兩個來源都失敗，回傳錯誤診斷
    return {'error': finmind_err}




# ─────────────────────────────────────────────
# 三大法人 30 天買賣超（FinMind）
# ─────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_three_institutions(ticker):
    """
    三大法人近 30 個交易日買賣超
    dataset: TaiwanStockInstitutionalInvestorsBuySell
    name 欄位：Foreign_Investor、Investment_Trust、Dealer_self、Dealer_Hedging、Foreign_Dealer_Self
    """
    if not ticker.endswith('.TW'):
        return None
    stock_code = ticker.replace('.TW', '')
    start_date = (datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d')
    try:
        url = (
            'https://api.finmindtrade.com/api/v4/data'
            f'?dataset=TaiwanStockInstitutionalInvestorsBuySell'
            f'&data_id={stock_code}&start_date={start_date}'
        )
        r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200 or r.json().get('status') != 200:
            return {'error': f'HTTP {r.status_code}'}
        rows = r.json().get('data', [])
        if not rows:
            return {'error': '無資料'}

        # 法人名稱對應
        GROUP = {
            'Foreign_Investor':    '外資',
            'Foreign_Dealer_Self': '外資',        # 合併入外資
            'Investment_Trust':    '投信',
            'Dealer_self':         '自營商',
            'Dealer_Hedging':      '自營商',       # 合併入自營商
        }
        from collections import defaultdict
        # {date: {法人: net_張}}
        daily = defaultdict(lambda: defaultdict(int))
        for row in rows:
            name   = row.get('name', '')
            group  = GROUP.get(name)
            if group is None:
                continue
            net = (int(row.get('buy', 0)) - int(row.get('sell', 0))) // 1000
            daily[row['date']][group] += net

        dates = sorted(daily.keys(), reverse=True)[:30]
        records = []
        for d in dates:
            rec = {'date': d}
            for g in ['外資', '投信', '自營商']:
                rec[g] = daily[d].get(g, 0)
            rec['合計'] = rec['外資'] + rec['投信'] + rec['自營商']
            records.append(rec)

        df_inst = pd.DataFrame(records).set_index('date')
        df_inst.index = pd.to_datetime(df_inst.index)
        df_inst = df_inst.sort_index()

        # 統計摘要（近 10 日）
        r10 = df_inst.tail(10)
        summary = {
            col: {
                'total':  int(r10[col].sum()),
                'consec': 0,
            }
            for col in ['外資', '投信', '自營商', '合計']
        }
        for col in ['外資', '投信', '自營商', '合計']:
            vals = df_inst[col].iloc[::-1].tolist()   # 最新在前
            cnt = 0
            for v in vals:
                if v > 0: cnt += 1
                else: break
            summary[col]['consec'] = cnt

        return {'df': df_inst, 'summary': summary}
    except Exception as e:
        return {'error': str(e)}


# ─────────────────────────────────────────────
# 融資融券餘額（FinMind）
# ─────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_margin(ticker):
    """
    融資融券近 60 日餘額
    dataset: TaiwanStockMarginPurchaseShortSale
    """
    if not ticker.endswith('.TW'):
        return None
    stock_code = ticker.replace('.TW', '')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    try:
        url = (
            'https://api.finmindtrade.com/api/v4/data'
            f'?dataset=TaiwanStockMarginPurchaseShortSale'
            f'&data_id={stock_code}&start_date={start_date}'
        )
        r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200 or r.json().get('status') != 200:
            return {'error': f'HTTP {r.status_code}'}
        rows = r.json().get('data', [])
        if not rows:
            return {'error': '無資料'}

        df_m = pd.DataFrame(rows)
        df_m['date'] = pd.to_datetime(df_m['date'])
        df_m = df_m.set_index('date').sort_index()

        # 欄位：MarginPurchaseBuy, MarginPurchaseSell, MarginPurchaseRedeem,
        #        MarginPurchaseTodayBalance, ShortSaleBuy, ShortSaleSell,
        #        ShortSaleTodayBalance 等
        num_cols = ['MarginPurchaseTodayBalance', 'ShortSaleTodayBalance',
                    'MarginPurchaseBuy', 'MarginPurchaseSell',
                    'ShortSaleBuy', 'ShortSaleSell']
        for c in num_cols:
            if c in df_m.columns:
                df_m[c] = pd.to_numeric(df_m[c], errors='coerce')

        df_m = df_m[[c for c in num_cols if c in df_m.columns]].dropna(how='all')

        last = df_m.iloc[-1] if not df_m.empty else pd.Series()
        prev = df_m.iloc[-2] if len(df_m) >= 2 else pd.Series()

        # 融資餘額趨勢（5日變化）
        margin_bal  = df_m['MarginPurchaseTodayBalance'] if 'MarginPurchaseTodayBalance' in df_m else pd.Series()
        short_bal   = df_m['ShortSaleTodayBalance']      if 'ShortSaleTodayBalance'      in df_m else pd.Series()
        margin_chg  = int(margin_bal.iloc[-1] - margin_bal.iloc[-5]) if len(margin_bal) >= 5 else 0
        short_chg   = int(short_bal.iloc[-1]  - short_bal.iloc[-5])  if len(short_bal)  >= 5 else 0

        # 融券覆蓋率 = 融券/融資（比值高 → 空方佔優）
        cover_ratio = (float(short_bal.iloc[-1]) / float(margin_bal.iloc[-1])
                       if (len(margin_bal) > 0 and float(margin_bal.iloc[-1]) > 0) else 0)

        return {
            'df':          df_m,
            'margin_bal':  margin_bal,
            'short_bal':   short_bal,
            'margin_chg':  margin_chg,
            'short_chg':   short_chg,
            'cover_ratio': cover_ratio,
        }
    except Exception as e:
        return {'error': str(e)}


# ─────────────────────────────────────────────
# 台指期貨三大法人未平倉淨額（FinMind）
# ─────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_futures_oi():
    """
    台指期貨三大法人未平倉多空淨口數（近 60 日）
    dataset: TaiwanFuturesInstitutionalInvestors
    商品：TX（台指期）
    """
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    try:
        url = (
            'https://api.finmindtrade.com/api/v4/data'
            f'?dataset=TaiwanFuturesInstitutionalInvestors'
            f'&data_id=TX&start_date={start_date}'
        )
        r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200 or r.json().get('status') != 200:
            return {'error': f'HTTP {r.status_code}'}
        rows = r.json().get('data', [])
        if not rows:
            return {'error': '無資料'}

        df_f = pd.DataFrame(rows)
        df_f['date'] = pd.to_datetime(df_f['date'])

        # FinMind 欄位：name / long_open_interest / short_open_interest /
        #               long_open_interest_balance / short_open_interest_balance
        GROUP = {
            '自營商':         '自營商',
            '投信':           '投信',
            '外資及大陸地區': '外資',
        }
        from collections import defaultdict
        daily_net = defaultdict(lambda: defaultdict(int))
        for _, row in df_f.iterrows():
            grp = GROUP.get(row.get('institutional_investors', ''))
            if grp is None:
                continue
            net = int(row.get('long_open_interest_balance_volume', 0) or 0)                 - int(row.get('short_open_interest_balance_volume', 0) or 0)
            daily_net[row['date']][grp] += net

        dates = sorted(daily_net.keys())
        records = []
        for d in dates:
            rec = {'date': d}
            for g in ['外資', '投信', '自營商']:
                rec[g] = daily_net[d].get(g, 0)
            rec['合計'] = rec['外資'] + rec['投信'] + rec['自營商']
            records.append(rec)

        df_oi = pd.DataFrame(records).set_index('date').sort_index()

        # 最新一日數值
        last = df_oi.iloc[-1] if not df_oi.empty else pd.Series()
        prev = df_oi.iloc[-2] if len(df_oi) >= 2 else pd.Series()
        chg  = {col: int(last.get(col, 0) - prev.get(col, 0))
                for col in ['外資', '投信', '自營商', '合計']}

        # 外資淨多口 > 0 → 偏多
        foreign_net = int(last.get('外資', 0))
        total_net   = int(last.get('合計', 0))
        bullish     = foreign_net > 0 and total_net > 0

        return {
            'df':          df_oi,
            'last':        last.to_dict(),
            'chg':         chg,
            'foreign_net': foreign_net,
            'total_net':   total_net,
            'bullish':     bullish,
        }
    except Exception as e:
        return {'error': str(e)}

# ─────────────────────────────────────────────
# 本益比河流圖資料（FinMind TaiwanStockPER）
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_per_river(ticker, years=5):
    """
    從 FinMind 抓歷史 PER + 每季 EPS，計算本益比河流圖帶狀線
    回傳 dict：
      per_df   : 每日 PER（DatetimeIndex，PER 欄）
      band_df  : 每日河流帶（index=日期，欄=各倍數股價）
      eps_df   : 每季 EPS（DatetimeIndex，EPS 欄）
      stats    : {'current_per', 'avg', 'std', 'pct_rank', 'zone'}
    """
    if not ticker.endswith('.TW'):
        return None
    stock_code = ticker.replace('.TW', '')
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')

    try:
        # ── 每日 PER ──────────────────────────────────────────────────────────
        url_per = (
            'https://api.finmindtrade.com/api/v4/data'
            f'?dataset=TaiwanStockPER&data_id={stock_code}&start_date={start_date}'
        )
        r = requests.get(url_per, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200 or r.json().get('status') != 200:
            return {'error': f'PER API HTTP {r.status_code}'}
        per_rows = r.json().get('data', [])
        if not per_rows:
            return {'error': 'PER 資料空白'}

        per_df = pd.DataFrame(per_rows)
        per_df['date'] = pd.to_datetime(per_df['date'])
        per_df = per_df.set_index('date').sort_index()
        per_df['PER'] = pd.to_numeric(per_df['PER'], errors='coerce')
        # FinMind TaiwanStockPER 無 stock_price 欄位，只保留 PER
        per_df = per_df[['PER']].dropna()

        # ── 每季 EPS（用於計算河流帶）────────────────────────────────────────
        url_eps = (
            'https://api.finmindtrade.com/api/v4/data'
            f'?dataset=TaiwanStockFinancialStatements&data_id={stock_code}&start_date={start_date}'
        )
        r2 = requests.get(url_eps, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        eps_annual = None
        if r2.status_code == 200 and r2.json().get('status') == 200:
            fin_rows = r2.json().get('data', [])
            if fin_rows:
                fin_df = pd.DataFrame(fin_rows)
                # 取 EPS（type == 'EPS'）
                eps_rows = fin_df[fin_df['type'] == 'EPS'].copy()
                if not eps_rows.empty:
                    eps_rows['date'] = pd.to_datetime(eps_rows['date'])
                    eps_rows['value'] = pd.to_numeric(eps_rows['value'], errors='coerce')
                    eps_rows = eps_rows.set_index('date').sort_index()
                    # 用滾動 4 季 EPS 合計（TTM）
                    eps_annual = eps_rows['value'].resample('QE').sum()
                    eps_annual = eps_annual.rolling(4).sum()  # TTM EPS

        # ── 計算河流帶 ────────────────────────────────────────────────────────
        # 方法一：有 TTM EPS → 用固定 PE 倍數
        # 方法二：沒有 EPS → 從歷史 PER 分位數計算價格帶
        band_df   = pd.DataFrame(index=per_df.index)
        multiples = [10, 15, 20, 25, 30]
        band_colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']

        if eps_annual is not None and len(eps_annual.dropna()) >= 2:
            # 把季度 EPS 前向填充到每日
            daily_eps = eps_annual.reindex(per_df.index, method='ffill')
            for m in multiples:
                band_df[f'{m}x'] = daily_eps * m
            eps_latest = float(eps_annual.dropna().iloc[-1])
        else:
            # 備援：從歷史 PER × 目前股價反推隱含帶狀
            # 無股價資料備援：用 PER 百分位數推算各帶「合理價格」
            # 以近期 PER 均值 × 各分位推算（注意：band_df 填入的是「股價帶」需搭配外部股價）
            # 此備援模式帶狀線以「歷史平均 PER ± 倍數比例」表示，呈現意義為相對位置
            valid_per_vals = per_df['PER'].replace(0, np.nan).dropna()
            q_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
            q_pers = valid_per_vals.quantile(q_vals)
            # 這裡 band_df 先填空，交由 build_per_chart 用外部股價自行計算
            eps_latest = 0.0
            band_df = pd.DataFrame(index=per_df.index)  # 空的，由圖表函數處理

        band_df = band_df.dropna(how='all')

        # ── 統計（估值位置）─────────────────────────────────────────────────
        valid_per = per_df['PER'].replace(0, np.nan).dropna()
        cur_per   = float(valid_per.iloc[-1]) if len(valid_per) else np.nan
        avg_per   = float(valid_per.mean())
        std_per   = float(valid_per.std())
        pct_rank  = float((valid_per < cur_per).mean() * 100) if not np.isnan(cur_per) else np.nan

        if np.isnan(cur_per):
            zone = 'unknown'
        elif cur_per < avg_per - std_per:
            zone = 'cheap'       # 歷史低估
        elif cur_per < avg_per:
            zone = 'fair_low'    # 合理偏低
        elif cur_per < avg_per + std_per:
            zone = 'fair_high'   # 合理偏高
        else:
            zone = 'expensive'   # 歷史高估

        return {
            'per_df':      per_df,
            'band_df':     band_df,
            'multiples':   multiples,
            'band_colors': band_colors,
            'eps_latest':  eps_latest,
            'stats': {
                'current_per': cur_per,
                'avg':         avg_per,
                'std':         std_per,
                'pct_rank':    pct_rank,
                'zone':        zone,
            },
        }
    except Exception as e:
        return {'error': str(e)}


def build_per_chart(per_data, ticker, price_series=None):
    """建立本益比河流圖（Plotly）— 完整重寫，移除 fillcolor 格式問題"""

    def hex_to_rgba(hex_c, alpha=0.15):
        h = hex_c.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    per_df    = per_data['per_df']
    band_df   = per_data.get('band_df', pd.DataFrame()).copy()
    multiples = per_data['multiples']
    colors    = per_data['band_colors']

    # 備援：band_df 空的時候用外部股價反推隱含 EPS 建帶
    if band_df.empty and price_series is not None:
        price_al  = price_series.reindex(per_df.index, method='ffill')
        imp_eps   = (price_al / per_df['PER'].replace(0, np.nan)).rolling(60, min_periods=1).median()
        for m in multiples:
            band_df[f'{m}x'] = imp_eps * m
        band_df = band_df.dropna(how='all')

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.65, 0.35],
        subplot_titles=['本益比河流圖（股價 vs PE 倍數帶）', '歷史本益比（PER）']
    )

    # ── 河流帶：由低倍數到高倍數，相鄰兩帶之間 fill='tonexty' ──────────────
    band_cols = [f'{m}x' for m in multiples if f'{m}x' in band_df.columns]
    for i, col in enumerate(band_cols):
        y = band_df[col].dropna()
        if y.empty:
            continue
        color = colors[i]
        if i == 0:
            # 最低帶：無填充，只畫線
            fig.add_trace(go.Scatter(
                x=y.index, y=y,
                name=f'PE {multiples[i]}x',
                mode='lines',
                line=dict(color=color, width=1),
                showlegend=True,
            ), row=1, col=1)
        else:
            # 與前一帶之間填色
            fig.add_trace(go.Scatter(
                x=y.index, y=y,
                name=f'PE {multiples[i]}x',
                mode='lines',
                line=dict(color=color, width=1),
                fill='tonexty',
                fillcolor=hex_to_rgba(color, alpha=0.18),
                showlegend=True,
            ), row=1, col=1)

    # ── 股價線 ──────────────────────────────────────────────────────
    if price_series is not None:
        price_plot = price_series.reindex(per_df.index, method='ffill').dropna()
        if not price_plot.empty:
            fig.add_trace(go.Scatter(
                x=price_plot.index, y=price_plot,
                name='股價',
                mode='lines',
                line=dict(color='#ffffff', width=2.5),
                hovertemplate='%{x|%Y-%m-%d}<br>股價：%{y:.1f}<extra></extra>',
            ), row=1, col=1)

    # ── 歷史 PER ────────────────────────────────────────────────────
    valid = per_df['PER'].replace(0, np.nan).dropna()
    avg   = float(valid.mean())
    std   = float(valid.std())

    fig.add_trace(go.Scatter(
        x=valid.index, y=valid,
        name='PER',
        mode='lines',
        line=dict(color='#45aaf2', width=1.5),
        hovertemplate='%{x|%Y-%m-%d}<br>PER：%{y:.1f}x<extra></extra>',
    ), row=2, col=1)

    for level, color, dash, label in [
        (avg + std, '#e74c3c', 'dash', f'+1σ  {avg+std:.1f}x'),
        (avg,       '#f39c12', 'dot',  f'均值 {avg:.1f}x'),
        (avg - std, '#2ecc71', 'dash', f'-1σ  {avg-std:.1f}x'),
    ]:
        fig.add_hline(
            y=level, row=2, col=1,
            line=dict(color=color, width=1, dash=dash),
            annotation_text=label,
            annotation_position='right',
            annotation_font=dict(size=10, color=color),
        )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        height=620, hovermode='x unified',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02, x=0, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=60, r=100, t=50, b=40),
        font=dict(color='#c9d1d9', size=11),
    )
    fig.update_yaxes(gridcolor='#21262d')
    fig.update_xaxes(gridcolor='#21262d')
    fig.update_yaxes(title_text='股價 (TWD)', row=1, col=1)
    fig.update_yaxes(title_text='PER (倍)',   row=2, col=1)
    return fig

# ─────────────────────────────────────────────
# 計算技術指標
# ─────────────────────────────────────────────
def calc_indicators(df, cfg):
    df = df.copy()
    ma_periods  = cfg['ma']
    vol_ma_n    = cfg['vol_ma']
    trigger_n   = cfg['trigger_ma']
    slope_n     = cfg['slope_periods']
    support_n   = cfg['support_periods']

    # 均線
    for p in ma_periods:
        df[f'MA{p}'] = df['Close'].rolling(p).mean()
    # 觸發均線（若與 ma_periods 重複也沒關係）
    df[f'MA{trigger_n}'] = df['Close'].rolling(trigger_n).mean()

    # 季線斜率（最短均線的前N期變化）
    short_ma_col = f'MA{ma_periods[0]}'
    df['MA_slope'] = df[short_ma_col] - df[short_ma_col].shift(slope_n)

    # RSI(14)
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    # 布林通道(20)
    df['BB_mid']   = df['Close'].rolling(20).mean()
    std            = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * std
    df['BB_lower'] = df['BB_mid'] - 2 * std
    df['BB_pct']   = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # KD(9,3,3)
    low9  = df['Low'].rolling(9).min()
    high9 = df['High'].rolling(9).max()
    rsv   = (df['Close'] - low9) / (high9 - low9 + 1e-9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    # 成交量均線
    df[f'Vol_MA{vol_ma_n}'] = df['Volume'].rolling(vol_ma_n).mean()

    # 結構支撐：前N期最低點
    df['Support'] = df['Low'].rolling(support_n).min()
    df['Near_support'] = (
        ((df['Close'] - df['Support']).abs() / df['Support']) <= 0.03
    )

    # 量能雙條件：
    #   前期量萎縮（近3期均量 < 量均線）
    #   + 當期量增（當期 > 上一期）
    vol_col = f'Vol_MA{vol_ma_n}'
    df['Vol_shrink_recent'] = df['Volume'].rolling(3).mean().shift(1) < df[vol_col].shift(1)
    df['Vol_expanding']     = df['Volume'] > df['Volume'].shift(1)
    df['Vol_dual']          = df['Vol_shrink_recent'] & df['Vol_expanding']

    # MACD 柱縮短（負值收斂，提前卡位）
    df['MACD_shortening'] = (
        (df['MACD_hist'] < 0) &
        (df['MACD_hist'].abs() < df['MACD_hist'].shift(1).abs()) &
        (df['MACD_hist'].shift(1) < 0)
    )

    # 進場觸發：突破前期高點 OR 站上短期均線
    df['Price_trigger'] = (
        (df['Close'] > df['High'].shift(1)) |
        (df['Close'] > df[f'MA{trigger_n}'])
    )

    return df

# ─────────────────────────────────────────────
# 訊號偵測（10分制）
# ─────────────────────────────────────────────
def detect_signals(df, cfg, rsi_low, score_th):
    df = df.copy()
    ma_periods  = cfg['ma']
    vol_ma_col  = f'Vol_MA{cfg["vol_ma"]}'
    ma_annual   = f'MA{ma_periods[-1]}'   # 年線
    ma_short    = f'MA{ma_periods[0]}'    # 最短均線

    # ── 趨勢（2分）──────────────────────────────
    t1 = df['Close'] > df[ma_annual]        # 收盤 > 年線
    t2 = df['MA_slope'] > 0                 # 季線斜率向上

    # ── 動能（3分）──────────────────────────────
    m1 = df['RSI'] < rsi_low                # RSI 超賣
    m2 = df['K'] < 25                       # KD 超賣
    m3 = df['MACD_shortening']              # MACD柱縮短（提前卡位）

    # ── 結構（3分）──────────────────────────────
    s1 = df['Close'] <= df['BB_lower'] * 1.02   # 接近布林下軌
    s2 = df['Near_support']                      # 接近結構支撐 ±3%
    s3 = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))  # KD黃金交叉

    # ── 量能（1分）──────────────────────────────
    v1 = df['Vol_dual']                     # 下跌量縮 + 反彈量增

    # ── 觸發（1分）──────────────────────────────
    tr = df['Price_trigger']                # 突破前高 or 站上短期均線

    # 儲存各條件結果（供說明用）
    for col, cond in zip(['cond_t1','cond_t2','cond_m1','cond_m2','cond_m3',
                          'cond_s1','cond_s2','cond_s3','cond_v1','cond_tr'],
                         [t1, t2, m1, m2, m3, s1, s2, s3, v1, tr]):
        df[col] = cond

    score = (t1.astype(int) + t2.astype(int) +
             m1.astype(int) + m2.astype(int) + m3.astype(int) +
             s1.astype(int) + s2.astype(int) + s3.astype(int) +
             v1.astype(int) + tr.astype(int))

    df['signal_score']  = score
    df['signal_strong'] = score >= score_th
    df['signal_medium'] = score == (score_th - 1)
    df['trend_bearish'] = ~(t1 | t2)  # 兩個趨勢條件都不滿足 → 趨勢偏空

    return df

# ─────────────────────────────────────────────
# 建立圖表
# ─────────────────────────────────────────────
def build_chart(df, cfg, interval_label, rsi_low, use_trend_filter):
    ma_periods  = cfg['ma']
    vol_ma_col  = f'Vol_MA{cfg["vol_ma"]}'
    trigger_n   = cfg['trigger_ma']

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
        line=dict(color='rgba(200,200,200,0.22)', width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'],
        fill='tonexty', fillcolor='rgba(173,216,230,0.08)',
        line=dict(color='rgba(200,200,200,0.22)', width=1), name='布林通道'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_mid'],
        line=dict(color='#8888aa', width=1, dash='dot'), showlegend=False), row=1, col=1)

    # 結構支撐線
    fig.add_trace(go.Scatter(x=df.index, y=df['Support'],
        line=dict(color='rgba(255,200,80,0.45)', width=1.2, dash='dash'),
        name=f'結構支撐({cfg["support_periods"]}期低點)'), row=1, col=1)

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

    # 觸發均線
    fig.add_trace(go.Scatter(x=df.index, y=df[f'MA{trigger_n}'],
        name=f'觸發線(MA{trigger_n})',
        line=dict(color='rgba(170,255,170,0.6)', width=1, dash='dot')), row=1, col=1)

    # 強訊號（通過趨勢濾網）
    strong_ok = df[df['signal_strong'] & (~df['trend_bearish'] | ~use_trend_filter)]
    if not strong_ok.empty:
        fig.add_trace(go.Scatter(
            x=strong_ok.index, y=strong_ok['Low'] * 0.974, mode='markers+text',
            name='低點訊號★',
            marker=dict(symbol='triangle-up', size=14, color='#00ff88',
                        line=dict(color='white', width=1)),
            text=['★'] * len(strong_ok), textposition='bottom center',
            textfont=dict(size=10, color='#00ff88'),
            hovertemplate=(
                '<b>低點訊號</b><br>日期：%{x|%Y-%m-%d}<br>'
                '收盤：%{customdata[0]:.1f}<br>評分：%{customdata[1]}/10<extra></extra>'
            ),
            customdata=np.stack([strong_ok['Close'], strong_ok['signal_score']], axis=-1)
        ), row=1, col=1)

    # 強訊號但趨勢偏空（灰色警示）
    if use_trend_filter:
        strong_bearish = df[df['signal_strong'] & df['trend_bearish']]
        if not strong_bearish.empty:
            fig.add_trace(go.Scatter(
                x=strong_bearish.index, y=strong_bearish['Low'] * 0.974, mode='markers',
                name='訊號??（空頭濾除）',
                marker=dict(symbol='triangle-up', size=11,
                            color='rgba(160,160,160,0.45)',
                            line=dict(color='white', width=1)),
                hovertemplate=(
                    '<b>訊號（趨勢偏空，謹慎）</b><br>日期：%{x|%Y-%m-%d}<br>'
                    '評分：%{customdata[0]}/10<extra></extra>'
                ),
                customdata=np.column_stack([strong_bearish['signal_score'].values])
            ), row=1, col=1)

    # 中等訊號
    medium = df[df['signal_medium'] & ~df['signal_strong']]
    if not medium.empty:
        fig.add_trace(go.Scatter(
            x=medium.index, y=medium['Low'] * 0.974, mode='markers',
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

    # MACD（縮短的柱標特別標記）
    hist_colors = ['#ff4444' if v >= 0 else '#22bb55' for v in df['MACD_hist']]
    # MACD shortening bars: slightly lighter color
    for i, (shortening, color) in enumerate(zip(df['MACD_shortening'], hist_colors)):
        if shortening:
            hist_colors[i] = '#88ddaa'  # 特別標記「縮短中」的柱
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD柱',
        marker_color=hist_colors, opacity=0.75), row=3, col=1)
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
# 底部綜合說明（升級版）
# ─────────────────────────────────────────────
def render_analysis(df, cfg, interval_label, ticker, rsi_low, score_th,
                    use_trend_filter, market_env, institutional):
    last       = df.iloc[-1]
    prev       = df.iloc[-2]
    score      = int(last['signal_score'])
    bar_unit   = cfg['bar_unit']
    prev_unit  = cfg['prev_unit']
    ma_periods = cfg['ma']
    vol_ma_col = f'Vol_MA{cfg["vol_ma"]}'
    trigger_n  = cfg['trigger_ma']
    trend_bearish = bool(last['trend_bearish'])

    st.markdown("---")
    st.subheader("?? 升級版四維技術分析（10分制）")

    # ── 整體研判 ──────────────────────────────────
    effective = score - (2 if trend_bearish and use_trend_filter else 0)
    if effective >= score_th:
        icon, label = "??", "偏多（低點訊號明顯）"
        desc = f"訊號評分 **{score}/10**，多項指標共振。"
        if trend_bearish and use_trend_filter:
            desc += " ?? **趨勢偏弱，訊號可靠度下降，建議輕倉試探或等待趨勢確認。**"
    elif effective >= score_th - 2:
        icon, label = "??", "中性偏多（弱訊號）"
        desc = f"訊號評分 **{score}/10**，訊號強度中等，建議等待更多確認後操作。"
    elif last['RSI'] > 65 or last['K'] > 75:
        icon, label = "??", "偏空（指標偏高）"
        desc = f"評分 **{score}/10**，多項指標處於高檔，留意回調風險。"
    else:
        icon, label = "?", "中性（無明顯訊號）"
        desc = f"評分 **{score}/10**，各指標中性，建議觀望。"

    st.markdown(f"### {icon} 整體研判：{label}")
    st.info(desc)

    # ── 四維評分儀表板 ────────────────────────────
    t_score = int(last['cond_t1']) + int(last['cond_t2'])
    m_score = int(last['cond_m1']) + int(last['cond_m2']) + int(last['cond_m3'])
    s_score = int(last['cond_s1']) + int(last['cond_s2']) + int(last['cond_s3'])
    v_score = int(last['cond_v1']) + int(last['cond_tr'])

    def s_icon(s, mx): return "??" if s == mx else ("??" if s > 0 else "??")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(f"{s_icon(t_score,2)} 趨勢",  f"{t_score} / 2",  "年線+斜率")
    with col2: st.metric(f"{s_icon(m_score,3)} 動能",  f"{m_score} / 3",  "RSI+KD+MACD縮短")
    with col3: st.metric(f"{s_icon(s_score,3)} 結構",  f"{s_score} / 3",  "布林+支撐+KD交叉")
    with col4: st.metric(f"{s_icon(v_score,2)} 量能+觸發", f"{v_score} / 2", "量雙條件+突破觸發")

    st.markdown("")

    # ── 詳細說明（左右兩欄）────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        # 趨勢
        st.markdown("**?? 趨勢濾網（避免空頭撿便宜）**")
        ma_annual_col   = f'MA{ma_periods[-1]}'
        ma_short_col    = f'MA{ma_periods[0]}'
        annual_label    = cfg['ma_labels'][-1].split('(')[0]
        short_label     = cfg['ma_labels'][0].split('(')[0]
        slope_v         = last['MA_slope']

        t1v = ("??" if last['cond_t1'] else "??")
        t2v = ("??" if last['cond_t2'] else "??")
        st.markdown(
            f"- {t1v} 收盤（{last['Close']:.1f}）"
            f"{'>' if last['cond_t1'] else '<'} {annual_label}（{last[ma_annual_col]:.1f}）"
            f"，{'多頭市場' if last['cond_t1'] else '空頭市場'}"
        )
        st.markdown(
            f"- {t2v} {short_label}斜率 {slope_v:+.1f}"
            f"，{'向上（趨勢上升）' if last['cond_t2'] else '向下（趨勢下降）'}"
        )
        if trend_bearish and use_trend_filter:
            st.warning("?? 兩項趨勢條件皆未通過，屬空頭環境，訊號勝率大幅下降")

        # 動能
        st.markdown("**?? 動能指標（Momentum）**")
        rsi_v = last['RSI']
        k_v, d_v = last['K'], last['D']
        h_now, h_prev = last['MACD_hist'], prev['MACD_hist']

        rsi_txt = (f"?? 超賣（{rsi_v:.1f}），反彈機率上升" if rsi_v < rsi_low else
                   f"?? 超買（{rsi_v:.1f}），留意回調" if rsi_v > 70 else
                   f"?? 中性（{rsi_v:.1f}）")
        kd_txt = (f"?? KD超賣（K:{k_v:.1f} D:{d_v:.1f}）" if k_v < 25 else
                  f"?? KD超買（K:{k_v:.1f}）" if k_v > 75 else
                  f"?? KD中性（K:{k_v:.1f} D:{d_v:.1f}）")

        if last['cond_m3']:
            macd_txt = f"?? MACD柱縮短（{h_prev:.2f}→{h_now:.2f}），即將翻正，提前卡位"
        elif h_now > 0 and h_prev <= 0:
            macd_txt = f"?? MACD柱{bar_unit}翻正（{h_now:.2f}），動能反轉（但可能稍晚）"
        elif h_now > 0:
            macd_txt = f"?? MACD柱持續為正（{h_now:.2f}），多頭延續"
        else:
            macd_txt = f"?? MACD柱為負（{h_now:.2f}），尚未縮短"

        st.markdown(f"- **RSI(14)**：{rsi_txt}")
        st.markdown(f"- **KD(9,3,3)**：{kd_txt}")
        st.markdown(f"- **MACD柱**：{macd_txt}")

    with col_b:
        # 結構
        st.markdown("**?? 結構指標（Price Structure）**")
        bb_pct   = last['BB_pct'] * 100
        support  = last['Support']
        dist_pct = abs(last['Close'] - support) / support * 100

        s1_txt = (f"?? 接近布林下軌（%B:{bb_pct:.1f}%，下軌{last['BB_lower']:.1f}）" if last['cond_s1'] else
                  f"?? 接近布林上軌（%B:{bb_pct:.1f}%）" if bb_pct > 80 else
                  f"?? 布林中段（%B:{bb_pct:.1f}%）")
        s2_txt = (f"?? 接近前{cfg['support_periods']}期低點支撐（距低點{dist_pct:.1f}%，支撐位{support:.1f}）"
                  if last['cond_s2'] else
                  f"?? 距支撐 {dist_pct:.1f}%（支撐位{support:.1f}），尚未到達")
        kd_cross = bool(last['K'] > last['D'] and prev['K'] <= prev['D'])
        s3_txt = (f"?? {bar_unit}KD黃金交叉（K:{k_v:.1f} D:{d_v:.1f}），動能反轉" if kd_cross else
                  f"?? K>D尚未交叉（K:{last['K']:.1f} D:{last['D']:.1f}）" if last['K'] > last['D'] else
                  f"?? KD死亡排列（K:{last['K']:.1f} < D:{last['D']:.1f}）")

        st.markdown(f"- **布林通道**：{s1_txt}")
        st.markdown(f"- **結構支撐**：{s2_txt}")
        st.markdown(f"- **KD交叉**：{s3_txt}")

        # 量能 + 觸發
        st.markdown("**?? 量能 + 進場觸發（Volume & Trigger）**")
        vol_ratio = last['Volume'] / last[vol_ma_col] if last[vol_ma_col] > 0 else 1
        v1_txt = (f"?? 下跌量縮後放量（{bar_unit}量為均量{vol_ratio*100:.0f}%），賣壓減弱+買盤進場"
                  if last['cond_v1'] else
                  f"?? 量能雙條件未達（{bar_unit}量為均量{vol_ratio*100:.0f}%），需觀察")
        tr_txt = (f"?? 突破前期高點（{prev['High']:.1f}）或站上MA{trigger_n}（{last[f'MA{trigger_n}']:.1f}），觸發進場"
                  if last['cond_tr'] else
                  f"?? 尚未突破前高（{prev['High']:.1f}）或MA{trigger_n}（{last[f'MA{trigger_n}']:.1f}），可等待")

        st.markdown(f"- **量能**：{v1_txt}")
        st.markdown(f"- **進場觸發**：{tr_txt}")

    # ── 市場環境 ───────────────────────────────────
    st.markdown("---")
    st.markdown("**?? 市場環境濾網（大盤 0050.TW）**")
    if market_env:
        env_icon = "??" if market_env['bullish'] else "??"
        at = "站上年線" if market_env['above_annual'] else "跌破年線"
        rsi_e = market_env['rsi']
        env_msg = "多頭環境，順勢操作有利" if market_env['bullish'] else "空頭環境，逆勢操作勝率偏低，建議謹慎"
        st.markdown(f"{env_icon} 大盤（0050.TW）{at}，RSI {rsi_e:.1f}，{env_msg}")
    else:
        st.caption("大盤環境資料暫時無法取得")

    # ── 外資籌碼 ───────────────────────────────────
    if ticker.endswith('.TW'):
        st.markdown("**?? 外資籌碼（近5日，TWSE）**")
        if institutional and isinstance(institutional, dict) and 'error' not in institutional:
            consec = institutional['consecutive_buy']
            total  = institutional['total_net_5d']
            latest = institutional['latest_net']
            if institutional['bullish']:
                st.markdown(
                    f"?? 外資連續 **{consec} 日**買超，近5日合計 {total:+,} 張，"
                    f"低檔回補訊號（主力入場）"
                )
            elif consec == 0:
                st.markdown(
                    f"?? 外資近期持續賣超，最新日 {latest:+,} 張，籌碼面偏空"
                )
            else:
                st.markdown(
                    f"?? 外資買超 {consec} 日（未達連3日門檻），持續觀察是否延續"
                )

            recs = institutional.get('records', [])
            if recs:
                inst_df = pd.DataFrame(recs)
                inst_df['方向'] = inst_df['foreign_net'].map(
                    lambda x: '?? 買超' if x > 0 else '?? 賣超'
                )
                inst_df['買賣超（張）'] = inst_df['foreign_net'].map(lambda x: f"{x:+,}")
                inst_df = inst_df[['date','方向','買賣超（張）']].rename(columns={'date':'日期'})
                st.dataframe(inst_df, width='stretch', hide_index=True)
        else:
            err_msg = institutional.get('error', '未知錯誤') if isinstance(institutional, dict) else '回傳格式異常'
            st.warning(f"?? 外資資料無法取得｜{err_msg}")

    # ── 完整明細（可展開）────────────────────────────
    with st.expander("?? 10項評分條件完整明細"):
        annual_lbl = cfg['ma_labels'][-1].split('(')[0]
        short_lbl  = cfg['ma_labels'][0].split('(')[0]
        sup_n      = cfg['support_periods']

        cond_list = [
            ("【趨勢1】收盤 > 年線",
             bool(last['cond_t1']),
             f"收盤={last['Close']:.1f}，{annual_lbl}={last[f'MA{ma_periods[-1]}']:.1f}"),
            ("【趨勢2】季線斜率向上",
             bool(last['cond_t2']),
             f"斜率={last['MA_slope']:+.1f}（{short_lbl}近{cfg['slope_periods']}期）"),
            ("【動能1】RSI < 超賣門檻",
             bool(last['cond_m1']),
             f"RSI={last['RSI']:.1f}，門檻={rsi_low}"),
            ("【動能2】K值 < 25",
             bool(last['cond_m2']),
             f"K={last['K']:.1f}"),
            ("【動能3】MACD柱縮短（負值收斂，提前卡位）",
             bool(last['cond_m3']),
             f"{bar_unit}={last['MACD_hist']:.2f}，{prev_unit}={prev['MACD_hist']:.2f}"),
            ("【結構1】接近布林下軌（%B < 20%）",
             bool(last['cond_s1']),
             f"收盤={last['Close']:.1f}，下軌={last['BB_lower']:.1f}，%B={last['BB_pct']*100:.1f}%"),
            (f"【結構2】接近前{sup_n}期低點支撐 ±3%",
             bool(last['cond_s2']),
             f"收盤={last['Close']:.1f}，支撐={last['Support']:.1f}，距離={dist_pct:.1f}%"),
            ("【結構3】KD 黃金交叉",
             bool(last['cond_s3']),
             f"K={last['K']:.1f} D={last['D']:.1f}（{bar_unit}）"),
            (f"【量能】下跌量縮 + {bar_unit}量增（雙條件）",
             bool(last['cond_v1']),
             f"{bar_unit}量={last['Volume']/1e8:.2f}億，均量={last[vol_ma_col]/1e8:.2f}億"),
            (f"【觸發】突破前期高點 或 站上MA{trigger_n}",
             bool(last['cond_tr']),
             f"收盤={last['Close']:.1f}，前高={prev['High']:.1f}，MA{trigger_n}={last[f'MA{trigger_n}']:.1f}"),
        ]

        for name, passed, detail in cond_list:
            st.markdown(f"{'?' if passed else '?'} **{name}**　*（{detail}）*")

        verdict = ("→ 強低點訊號" if score >= score_th else
                   "→ 觀察訊號"   if score >= score_th - 2 else
                   "→ 無明顯低點訊號")
        st.markdown(f"**合計：{score} / 10 項　{verdict}**")
        if trend_bearish and use_trend_filter:
            st.warning("趨勢偏弱（年線下方+季線下彎），建議降低倉位或等趨勢確認後再行動")

    st.markdown("> ?? **免責聲明**：以上分析純為技術面參考，不構成任何投資建議。股市有風險，投資需謹慎。")


# ─────────────────────────────────────────────
# 主流程（手動觸發）
# ─────────────────────────────────────────────
period     = cfg['period_map'].get(years, '3y')
vol_ma_col = f'Vol_MA{cfg["vol_ma"]}'

st.title(f"?? {ticker}　{interval_label} 技術分析儀表板 v2")

# ── 手動執行按鈕（側邊欄底部已設定，主畫面也放一顆）─────────
run_clicked = st.button("?? 執行分析", type="primary", help="設定好參數後點此開始分析")

# 用 session_state 記住「已執行過」，切換週期/代碼時需重新按
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
    st.session_state.df          = None
    st.session_state.market_env  = None
    st.session_state.institutional = None
    st.session_state.per_data = None
    st.session_state.three_inst = None
    st.session_state.margin    = None
    st.session_state.futures_oi = None

query_key = f"{ticker}|{interval_label}|{years}|{rsi_low}|{score_th}"

if run_clicked:
    st.session_state.last_query = query_key
    # 清除舊快取，確保抓到最新資料
    fetch_data.clear()
    fetch_market_env.clear()
    fetch_institutional.clear()

    with st.spinner("?? 正在抓取股價資料..."):
        try:
            df = fetch_data(ticker, cfg['interval'], period)
            if df.empty:
                st.error("無法取得資料，請確認股票代碼（台股加 .TW，如 2330.TW）")
                st.stop()
            df = calc_indicators(df, cfg)
            df = detect_signals(df, cfg, rsi_low, score_th)
            st.session_state.df = df
        except Exception as e:
            st.error(f"資料抓取失敗：{e}")
            st.stop()

    with st.spinner("?? 抓取大盤環境與外資資料（FinMind）..."):
        st.session_state.market_env    = fetch_market_env(cfg['interval'])
        st.session_state.institutional = fetch_institutional(ticker) if ticker.endswith('.TW') else None

    with st.spinner('?? 抓取本益比歷史資料（FinMind）...'):
        st.session_state.per_data = fetch_per_river(ticker) if ticker.endswith('.TW') else None

    with st.spinner('?? 抓取三大法人、融資融券、期貨法人資料...'):
        if ticker.endswith('.TW'):
            st.session_state.three_inst  = fetch_three_institutions(ticker)
            st.session_state.margin      = fetch_margin(ticker)
        else:
            st.session_state.three_inst  = None
            st.session_state.margin      = None
        st.session_state.futures_oi = fetch_futures_oi()

elif st.session_state.df is None:
    st.info("?? 請在左側設定股票代碼與週期，再按「執行分析」開始")
    st.stop()
elif st.session_state.last_query != query_key:
    # 設定已變更（如切換週期），舊 df 的欄位不符，必須重新執行
    st.warning(
        f"?? 設定已變更（{st.session_state.last_query} → {query_key}），"
        "請按「執行分析」重新載入資料。"
    )
    st.stop()

df            = st.session_state.df
market_env    = st.session_state.market_env
institutional = st.session_state.institutional
per_data      = st.session_state.get('per_data')
three_inst    = st.session_state.get('three_inst')
margin_data   = st.session_state.get('margin')
futures_oi    = st.session_state.get('futures_oi')

last = df.iloc[-1]
trend_ok = bool(last['cond_t1']) or bool(last['cond_t2'])
market_ok = market_env is not None and market_env['bullish']
score_now = int(last['signal_score'])

# ── 指標卡片 ──────────────────────────────────
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
with c1:
    st.metric("收盤價", f"{last['Close']:.1f}",
              f"{last['Close'] - df.iloc[-2]['Close']:+.1f}")
with c2:
    st.metric("RSI(14)", f"{last['RSI']:.1f}",
              "超賣↑" if last['RSI'] < rsi_low else ("超買↓" if last['RSI'] > 70 else "中性"))
with c3:
    st.metric("KD", f"{last['K']:.0f} / {last['D']:.0f}",
              "超賣" if last['K'] < 25 else ("超買" if last['K'] > 75 else ""))
with c4:
    st.metric("趨勢濾網",
              "? 多頭" if trend_ok else "?? 空頭", "")
with c5:
    st.metric("大盤環境",
              "? 有利" if market_ok else ("?? 不利" if market_env else "—"), "")
with c6:
    st.metric("訊號評分", f"{score_now} / 10",
              "★ 強訊號" if last['signal_strong'] else ("△ 觀察" if last['signal_medium'] else ""))
with c7:
    if per_data and 'stats' in per_data:
        ps = per_data['stats']
        zone_map = {'cheap':'?? 低估','fair_low':'?? 合理偏低','fair_high':'?? 合理偏高','expensive':'?? 高估','unknown':'—'}
        st.metric("本益比 PER", f"{ps['current_per']:.1f}x" if not np.isnan(ps['current_per']) else "—",
                  zone_map.get(ps['zone'],'—'))
    else:
        st.metric("本益比 PER", "—", "")

# ── 圖表 ──────────────────────────────────────
fig = build_chart(df, cfg, interval_label, rsi_low, use_trend_filter)
st.plotly_chart(fig, width='stretch')

# ── 近期訊號表 ─────────────────────────────────
strong_signals = df[df['signal_strong']].tail(5)
if not strong_signals.empty:
    st.subheader(f"?? 近期強低點訊號（{interval_label}）")
    disp = strong_signals[['Close','RSI','K','D','BB_pct','signal_score','trend_bearish']].copy()
    disp.index = disp.index.strftime(cfg['date_fmt'])
    disp.columns = ['收盤價','RSI','K值','D值','BB%B','評分(/10)','趨勢偏弱']
    disp['BB%B']    = (disp['BB%B'] * 100).round(1)
    disp['趨勢偏弱'] = disp['趨勢偏弱'].map({True: '?? 是', False: '? 否'})
    st.dataframe(disp.style.format({
        '收盤價':'{:.1f}', 'RSI':'{:.1f}', 'K值':'{:.1f}',
        'D值':'{:.1f}', 'BB%B':'{:.1f}%', '評分(/10)':'{:.0f}'
    }), width='stretch')

st.caption(
    f"? 資料更新：{df.index[-1].strftime(cfg['date_fmt'])}  |  "
    f"週期：{interval_label}  |  ?? 僅供技術分析參考"
)

# ── 本益比河流圖 ───────────────────────────────
st.markdown("---")
st.subheader("?? 本益比河流圖（PER River Chart）")
if per_data and 'stats' in per_data:
    ps = per_data['stats']
    zone_label = {'cheap':'?? 歷史低估區','fair_low':'?? 合理偏低','fair_high':'?? 合理偏高','expensive':'?? 歷史高估區','unknown':'—'}
    zone_color = {'cheap':'success','fair_low':'info','fair_high':'warning','expensive':'error','unknown':'info'}

    pa, pb, pc, pd_ = st.columns(4)
    with pa: st.metric("當前 PER",  f"{ps['current_per']:.1f}x" if not np.isnan(ps['current_per']) else "—")
    with pb: st.metric("歷史均值",  f"{ps['avg']:.1f}x")
    with pc: st.metric("均值±1σ",   f"{ps['avg']-ps['std']:.1f}x ～ {ps['avg']+ps['std']:.1f}x")
    with pd_: st.metric("估值位置", f"高於歷史 {ps['pct_rank']:.0f}%" if not np.isnan(ps['pct_rank']) else "—",
                        zone_label.get(ps['zone'], '—'))

    zone_msgs = {
        'cheap':     '目前 PER 低於歷史均值 1 個標準差，屬歷史**低估**區間，長線佈局勝率較高。',
        'fair_low':  '目前 PER 低於歷史均值，屬合理偏低區間，估值具吸引力。',
        'fair_high': '目前 PER 高於歷史均值但未超過 +1σ，屬合理偏高，需留意獲利成長能否支撐。',
        'expensive': '目前 PER 超過歷史均值 +1σ，屬歷史**高估**區間，投資人已反映較高成長預期，回調風險上升。',
        'unknown':   'PER 資料不足，無法判斷估值區間。',
    }
    st.info(zone_msgs.get(ps['zone'], ''))

    # 取得日K收盤價（用 yfinance，確保頻率是日線）
    try:
        _t = yf.Ticker(ticker)
        _price_df = _t.history(period='5y', interval='1d', auto_adjust=True)
        _price_df.index = _price_df.index.tz_localize(None)
        price_series_for_per = _price_df['Close']
    except Exception:
        price_series_for_per = None
    per_fig = build_per_chart(per_data, ticker, price_series=price_series_for_per)
    st.plotly_chart(per_fig, width='stretch')

    st.caption(
        f"河流帶說明：PE 10x / 15x / 20x / 25x / 30x 對應隱含股價帶。"
        f"股價在低 PE 帶表示估值偏低，在高 PE 帶表示市場給予較高溢價。"
        f"資料來源：FinMind（TaiwanStockPER）"
    )
elif per_data and 'error' in per_data:
    st.caption(f"PER 河流圖暫時無法載入：{per_data['error']}")
elif not ticker.endswith('.TW'):
    st.caption("PER 河流圖僅支援台灣上市股票（代碼需以 .TW 結尾）")


# ══════════════════════════════════════════════
# 籌碼面綜合分析（三大法人 / 融資融券 / 期貨法人）
# ══════════════════════════════════════════════
st.markdown("---")
st.subheader("?? 籌碼面綜合分析")

tab1, tab2, tab3 = st.tabs(["?? 三大法人 30日買賣超", "?? 融資融券餘額", "?? 台指期貨法人未平倉"])

# ── Tab1：三大法人 ──────────────────────────────────────────────────
with tab1:
    if three_inst and 'df' in three_inst:
        ti_df  = three_inst['df']
        ti_sum = three_inst['summary']

        # 摘要卡片
        s1, s2, s3, s4 = st.columns(4)
        for col_w, key, label in [
            (s1, '外資',  '外資 近10日合計'),
            (s2, '投信',  '投信 近10日合計'),
            (s3, '自營商','自營商 近10日合計'),
            (s4, '合計',  '三大法人合計'),
        ]:
            val    = ti_sum[key]['total']
            consec = ti_sum[key]['consec']
            delta  = f"連{consec}日買超" if consec >= 2 else ("今日買超" if consec == 1 else "連續賣超")
            col_w.metric(label, f"{val:+,} 張", delta)

        st.markdown("")

        # 30 日折線圖
        fig_ti = go.Figure()
        color_map = {'外資':'#45aaf2', '投信':'#f5a623', '自營商':'#e056fd', '合計':'#00ff88'}
        for col in ['外資', '投信', '自營商', '合計']:
            if col in ti_df.columns:
                fig_ti.add_trace(go.Bar(
                    x=ti_df.index, y=ti_df[col], name=col,
                    marker_color=[('#ff4444' if v < 0 else color_map[col]) for v in ti_df[col]],
                    visible=True if col == '合計' else 'legendonly',
                ))
        fig_ti.update_layout(
            template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            height=320, barmode='group', hovermode='x unified',
            legend=dict(orientation='h', y=1.02),
            margin=dict(l=50, r=20, t=30, b=40),
            yaxis_title='買賣超（張）',
        )
        fig_ti.update_xaxes(gridcolor='#21262d')
        fig_ti.update_yaxes(gridcolor='#21262d', zeroline=True, zerolinecolor='#555')
        st.plotly_chart(fig_ti, width='stretch')

        # 研判說明
        foreign_10 = ti_sum['外資']['total']
        trust_10   = ti_sum['投信']['total']
        f_consec   = ti_sum['外資']['consec']
        t_consec   = ti_sum['投信']['consec']
        total_10   = ti_sum['合計']['total']

        if foreign_10 > 0 and trust_10 > 0:
            st.success(f"?? 外資與投信**同步買超**（外資近10日 {foreign_10:+,}張，投信 {trust_10:+,}張），"
                       f"籌碼面多頭訊號強烈，尤其外資連 {f_consec} 日買超，可信度較高。")
        elif foreign_10 > 0 and trust_10 <= 0:
            st.info(f"?? 外資買超（{foreign_10:+,}張）但投信賣超（{trust_10:+,}張），"
                    f"法人看法分歧，建議觀察外資是否持續。")
        elif foreign_10 <= 0 and trust_10 > 0:
            st.info(f"?? 投信買超（{trust_10:+,}張）但外資賣超（{foreign_10:+,}張），"
                    f"投信資金規模較小，整體籌碼仍偏弱。")
        else:
            st.warning(f"?? 外資（{foreign_10:+,}張）與投信（{trust_10:+,}張）**同步賣超**，"
                       f"法人持續出脫，籌碼面偏空。")

        # 明細表（最近10日）
        with st.expander("?? 近30日明細"):
            disp_ti = ti_df.copy()
            disp_ti.index = disp_ti.index.strftime('%Y/%m/%d')
            st.dataframe(
                disp_ti.sort_index(ascending=False).style.map(
                    lambda v: 'color: #ff6b6b' if v < 0 else 'color: #51cf66',
                    subset=['外資','投信','自營商','合計']
                ).format("{:+,}"),
                width='stretch'
            )
    elif three_inst and 'error' in three_inst:
        st.caption(f"三大法人資料無法取得：{three_inst['error']}")
    else:
        st.caption("三大法人資料僅支援台股（代碼需以 .TW 結尾）")

# ── Tab2：融資融券 ──────────────────────────────────────────────────
with tab2:
    if margin_data and 'df' in margin_data:
        mg_df  = margin_data['df']
        mb     = margin_data['margin_bal']
        sb     = margin_data['short_bal']
        mg_chg = margin_data['margin_chg']
        sg_chg = margin_data['short_chg']
        cover  = margin_data['cover_ratio']

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            val = int(mb.iloc[-1]) if not mb.empty else 0
            st.metric("融資餘額（張）", f"{val:,}", f"近5日 {mg_chg:+,}")
        with m2:
            val = int(sb.iloc[-1]) if not sb.empty else 0
            st.metric("融券餘額（張）", f"{val:,}", f"近5日 {sg_chg:+,}")
        with m3:
            st.metric("融券/融資 覆蓋率", f"{cover*100:.1f}%",
                      "空方佔優" if cover > 0.3 else "多方佔優")
        with m4:
            # 融資增 + 股價漲 → 正常多頭；融資減 + 股價漲 → 籌碼乾淨
            if mg_chg < 0:
                st.metric("籌碼狀態", "?? 融資減少", "籌碼趨於乾淨")
            elif mg_chg > 0:
                st.metric("籌碼狀態", "?? 融資增加", "留意追高風險")
            else:
                st.metric("籌碼狀態", "? 持平", "")

        # 雙軸圖：融資餘額 + 融券餘額
        fig_mg = make_subplots(specs=[[{"secondary_y": True}]])
        if not mb.empty:
            fig_mg.add_trace(go.Scatter(
                x=mb.index, y=mb, name='融資餘額',
                line=dict(color='#ff6b6b', width=2),
                fill='tozeroy', fillcolor='rgba(255,107,107,0.08)',
            ), secondary_y=False)
        if not sb.empty:
            fig_mg.add_trace(go.Bar(
                x=sb.index, y=sb, name='融券餘額',
                marker_color='rgba(69,170,242,0.55)',
            ), secondary_y=True)

        fig_mg.update_layout(
            template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            height=320, hovermode='x unified',
            legend=dict(orientation='h', y=1.02),
            margin=dict(l=60, r=60, t=30, b=40),
        )
        fig_mg.update_yaxes(title_text='融資餘額（張）', gridcolor='#21262d', secondary_y=False)
        fig_mg.update_yaxes(title_text='融券餘額（張）', gridcolor='#21262d', secondary_y=True)
        fig_mg.update_xaxes(gridcolor='#21262d')
        st.plotly_chart(fig_mg, width='stretch')

        # 研判說明
        if mg_chg < 0 and sg_chg <= 0:
            st.success("?? 融資減少且融券未明顯增加，籌碼逐漸乾淨，若股價同步止跌，"
                       "為**吸籌訊號**，低點可信度提升。")
        elif mg_chg < 0 and sg_chg > 0:
            st.warning("?? 融資減少同時融券增加（軋空佈局），市場分歧加大，"
                       "需觀察誰先認輸。若股價上漲，融券回補可能帶來短線急漲。")
        elif mg_chg > 0 and sg_chg > 0:
            st.info("?? 融資與融券同步增加，多空雙方皆在加碼，方向待確認，"
                    "避免追高或追空。")
        elif mg_chg > 0 and sg_chg <= 0:
            st.info("?? 融資增加，散戶追多積極，需留意過度槓桿風險。"
                    "若外資同步買超則訊號較可信，否則需謹慎。")

        st.caption(f"融券覆蓋率（{cover*100:.1f}%）：融券張數 ÷ 融資張數。"
                   "超過 30% 代表空頭部位相對重，軋空行情機率上升。")

    elif margin_data and 'error' in margin_data:
        st.caption(f"融資融券資料無法取得：{margin_data['error']}")
    else:
        st.caption("融資融券資料僅支援台股（代碼需以 .TW 結尾）")

# ── Tab3：台指期貨法人未平倉 ────────────────────────────────────────
with tab3:
    if futures_oi and 'df' in futures_oi:
        fo_df      = futures_oi['df']
        fo_last    = futures_oi['last']
        fo_chg     = futures_oi['chg']
        fo_foreign = futures_oi['foreign_net']
        fo_total   = futures_oi['total_net']

        f1, f2, f3, f4 = st.columns(4)
        for col_w, key, label in [
            (f1, '外資',  '外資 未平倉淨口'),
            (f2, '投信',  '投信 未平倉淨口'),
            (f3, '自營商','自營商 未平倉淨口'),
            (f4, '合計',  '三大法人合計'),
        ]:
            val = int(fo_last.get(key, 0))
            chg = fo_chg.get(key, 0)
            col_w.metric(label, f"{val:+,} 口",
                         f"增 {chg:+,} 口" if chg != 0 else "持平")

        # 折線圖
        fig_fo = go.Figure()
        color_map2 = {'外資':'#45aaf2', '投信':'#f5a623', '自營商':'#e056fd', '合計':'#00ff88'}
        for col in ['外資', '投信', '自營商', '合計']:
            if col in fo_df.columns:
                fig_fo.add_trace(go.Scatter(
                    x=fo_df.index, y=fo_df[col], name=col,
                    line=dict(color=color_map2[col], width=2),
                    mode='lines',
                    visible=True if col in ['外資', '合計'] else 'legendonly',
                ))
        fig_fo.add_hline(y=0, line=dict(color='#555', width=1))
        fig_fo.update_layout(
            template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            height=320, hovermode='x unified',
            legend=dict(orientation='h', y=1.02),
            margin=dict(l=50, r=20, t=30, b=40),
            yaxis_title='未平倉淨口數（口）',
        )
        fig_fo.update_xaxes(gridcolor='#21262d')
        fig_fo.update_yaxes(gridcolor='#21262d', zeroline=True, zerolinecolor='#555')
        st.plotly_chart(fig_fo, width='stretch')

        # 研判說明
        if fo_foreign > 0 and fo_total > 0:
            st.success(
                f"?? 外資台指期**淨多** {fo_foreign:+,} 口（近日變化 {fo_chg['外資']:+,} 口），"
                f"三大法人合計淨多 {fo_total:+,} 口。外資通常是方向性最強的法人，"
                f"多口佈局代表對後市看多。"
            )
        elif fo_foreign < 0 and fo_total < 0:
            st.error(
                f"?? 外資台指期**淨空** {fo_foreign:+,} 口（近日變化 {fo_chg['外資']:+,} 口），"
                f"三大法人合計淨空 {fo_total:+,} 口。法人整體偏空，"
                f"搭配現貨訊號需更謹慎。"
            )
        elif fo_foreign > 0 and fo_total < 0:
            st.info(
                f"?? 外資淨多 {fo_foreign:+,} 口，但自營商或投信拉低合計至 {fo_total:+,} 口。"
                f"法人看法分歧，以外資方向為主要參考。"
            )
        else:
            st.info(
                f"?? 外資淨空 {fo_foreign:+,} 口，但整體法人合計 {fo_total:+,} 口。"
                f"請搭配現貨外資動向綜合判斷。"
            )

        st.caption(
            "淨口數 = 多頭未平倉 - 空頭未平倉。正值代表法人看多台指，"
            "負值代表看空。外資期貨部位與現貨同向時，訊號可信度最高。"
            "資料來源：FinMind（TaiwanFuturesInstitutionalInvestors，商品代碼 TX）"
        )

        with st.expander("?? 近60日明細"):
            disp_fo = fo_df.copy()
            disp_fo.index = disp_fo.index.strftime('%Y/%m/%d')
            st.dataframe(
                disp_fo.sort_index(ascending=False).style.map(
                    lambda v: 'color: #ff6b6b' if v < 0 else 'color: #51cf66',
                    subset=['外資','投信','自營商','合計']
                ).format("{:+,}"),
                width='stretch'
            )
    elif futures_oi and 'error' in futures_oi:
        st.caption(f"期貨法人資料無法取得：{futures_oi['error']}")
    else:
        st.caption("資料載入中，請稍後...")


# ── 底部分析 ───────────────────────────────────
dist_pct = abs(last['Close'] - last['Support']) / last['Support'] * 100
render_analysis(
    df, cfg, interval_label, ticker, rsi_low, score_th,
    use_trend_filter, market_env, institutional
)