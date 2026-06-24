import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="BBW + MACD 金叉篩選器",
    page_icon="📊",
    layout="wide",
)

# ── 樣式 ──────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 320px; }
.block-container { padding-top: 1.5rem; }
.metric-label { font-size: 12px !important; }
.stDataFrame { font-size: 13px; }
div[data-testid="metric-container"] > div { gap: 2px; }
</style>
""", unsafe_allow_html=True)

# ── 預設清單 ──────────────────────────────────────
ALL_STOCKS = [
    "1101", "1102", "1103", "1104", "1108", "1109", "1110", "1210", "1213", "1215", "1216", "1218", "1220", "1227", "1229",
    "1231", "1232", "1233", "1301", "1303", "1304", "1305", "1306", "1308", "1309", "1310", "1312", "1313", "1314", "1315",
    "1402", "1404", "1409", "1413", "1414", "1416", "1417", "1418", "1419", "1434", "1436", "1440", "1441", "1442", "1443",
    "1444", "1445", "1446", "1447", "1448", "1449", "1503", "1504", "1506", "1507", "1512", "1513", "1514", "1515", "1516",
    "1517", "1519", "1521", "1522", "1524", "1525", "1526", "1527", "1528", "1529", "1537", "1538", "1540", "1541", "1542",
    "1543", "1544", "1545", "1546", "1603", "1605", "1752", "1760", "2002", "2006", "2008", "2010", "2014", "2015", "2016",
    "2031", "2032", "2038", "2049", "2059", "2201", "2204", "2207", "2208", "2301", "2303", "2308", "2317", "2324", "2327",
    "2330", "2337", "2344", "2347", "2351", "2353", "2354", "2356", "2357", "2360", "2364", "2367", "2376", "2377", "2379",
    "2382", "2383", "2385", "2388", "2395", "2408", "2409", "2412", "2448", "2449", "2454", "2458", "2501", "2502", "2503",
    "2504", "2505", "2506", "2511", "2515", "2516", "2520", "2521", "2522", "2603", "2605", "2606", "2607", "2608", "2609",
    "2610", "2611", "2615", "2616", "2618", "2701", "2702", "2707", "2712", "2723", "2727", "2731", "2801", "2820", "2823",
    "2833", "2850", "2851", "2852", "2881", "2882", "2883", "2884", "2885", "2886", "2887", "2888", "2889", "2890", "2891",
    "2892", "2905", "2910", "2912", "2915", "3008", "3014", "3016", "3026", "3034", "3035", "3037", "3041", "3044", "3045",
    "3231", "3293", "3443", "3481", "3533", "3596", "3630", "3665", "3698", "3711", "4119", "4126", "4128", "4129", "4144",
    "4148", "4153", "4154", "4157", "4160", "4166", "4168", "4171", "4174", "4175", "4176", "4177", "4179", "4904", "4906",
    "4977", "5274", "5876", "5880", "6223", "6269", "6274", "6415", "6505", "6515", "6531", "6669", "6770", "8046"
]

PRESETS = {
    "半導體":       ["2330","2303","2308","2337","2344","2379","2454","3443","3711","6770","3034","6415","3035","6531","2385"],
    "金融股":       ["2881","2882","2883","2884","2885","2886","2887","2891","2892","5876","5880"],
    "伺服器":       ["2317","2382","2357","2376","6669","3231","2353","3045","4906","3630"],
    "全部 239 檔":  ALL_STOCKS,
}

# ── FinMind API ───────────────────────────────────
def fetch_finmind(stock_id: str, token: str, days: int = 180) -> pd.DataFrame | None:
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        "https://api.finmindtrade.com/api/v4/data"
        f"?dataset=TaiwanStockPrice"
        f"&data_id={stock_id}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&token={token}"
    )
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        j = res.json()
        if j.get("status") != 200 or not j.get("data"):
            return None
        df = pd.DataFrame(j["data"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["close"] = df["close"].astype(float)
        return df
    except Exception:
        return None

def test_api(token: str) -> tuple[bool, str]:
    df = fetch_finmind("2330", token, days=10)
    if df is not None and len(df) > 0:
        return True, f"✅ 驗證成功，取得 {len(df)} 筆 2330 資料"
    return False, "❌ 驗證失敗，請確認 Token 是否正確"

# ── 指標計算 ──────────────────────────────────────
def calc_ema(series: np.ndarray, period: int) -> np.ndarray:
    k = 2 / (period + 1)
    ema = np.empty(len(series) - period + 1)
    ema[0] = series[:period].mean()
    for i in range(1, len(ema)):
        ema[i] = series[period - 1 + i] * k + ema[i - 1] * (1 - k)
    return ema

def calc_bb(closes: np.ndarray, length: int, std_mult: float) -> dict | None:
    if len(closes) < length:
        return None
    window = closes[-length:]
    mean = window.mean()
    sd = window.std(ddof=0)
    upper = mean + std_mult * sd
    lower = mean - std_mult * sd
    bbw = (upper - lower) / mean if mean > 0 else None
    if bbw is None:
        return None
    return {"mean": mean, "upper": upper, "lower": lower, "bbw": bbw}

def calc_macd(closes: np.ndarray, fast: int, slow: int, signal: int) -> dict | None:
    if len(closes) < slow + signal:
        return None
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    offset = len(ema_fast) - len(ema_slow)
    dif = ema_fast[offset:] - ema_slow
    if len(dif) < signal:
        return None
    dea = calc_ema(dif, signal)
    dif_aligned = dif[len(dif) - len(dea):]
    hist = dif_aligned - dea
    return {"dif": dif, "dea": dea, "hist": hist}

def detect_golden_cross(dif: np.ndarray, dea: np.ndarray, lookback: int) -> bool:
    n = min(len(dif), len(dea))
    d = dif[-n:]
    e = dea[-n:]
    for i in range(max(1, n - lookback), n):
        if d[i] > e[i] and d[i - 1] <= e[i - 1]:
            return True
    return False

# ── 單股分析 ──────────────────────────────────────
def analyze_stock(sym: str, token: str, params: dict) -> dict | None:
    df = fetch_finmind(sym, token, days=params["fetch_days"])
    if df is None or len(df) < params["slow"] + params["signal"] + 5:
        return None

    closes = df["close"].values

    bb = calc_bb(closes, params["bb_length"], params["bb_std"])
    if bb is None:
        return None

    macd = calc_macd(closes, params["fast"], params["slow"], params["signal"])
    if macd is None:
        return None

    latest_dif  = macd["dif"][-1]
    latest_dea  = macd["dea"][-1]
    latest_hist = macd["hist"][-1]
    has_golden  = detect_golden_cross(macd["dif"], macd["dea"], params["lookback"])

    return {
        "代號":    sym,
        "收盤價":  round(closes[-1], 2),
        "BBW":     round(bb["bbw"], 4),
        "上軌":    round(bb["upper"], 2),
        "下軌":    round(bb["lower"], 2),
        "DIF":     round(latest_dif, 4),
        "DEA":     round(latest_dea, 4),
        "Hist":    round(latest_hist, 4),
        "bbw_ok":  bb["bbw"] < params["bb_threshold"],
        "golden":  has_golden,
        "dif_neg": latest_dif < 0,
        "hist_pos": latest_hist > 0,
    }

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.title("📊 BBW + MACD 篩選器")
    st.caption("FinMind API × 布林通道壓縮 × DIF 上穿 DEA")
    st.divider()

    # API
    st.subheader("🔑 FinMind API")
    token = st.text_input("API Token", type="password", placeholder="輸入你的 FinMind Token")
    if st.button("驗證連線", use_container_width=True):
        if token:
            ok, msg = test_api(token)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        else:
            st.warning("請先輸入 Token")
    st.caption("[前往 FinMind 免費註冊](https://finmindtrade.com)，免費版每天 600 次請求")
    st.divider()

    # 布林參數
    st.subheader("📐 布林通道參數")
    bb_length    = st.number_input("週期（N）", min_value=5, max_value=60, value=20)
    bb_std       = st.number_input("標準差倍數", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    bb_threshold = st.number_input("BBW 閾值（上限）", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    st.divider()

    # MACD 參數
    st.subheader("📈 MACD 參數")
    col1, col2 = st.columns(2)
    with col1:
        macd_fast   = st.number_input("快線 EMA", min_value=2, max_value=50, value=12)
        macd_slow   = st.number_input("慢線 EMA", min_value=5, max_value=100, value=26)
    with col2:
        macd_signal = st.number_input("Signal 線", min_value=2, max_value=30, value=9)
        lookback    = st.number_input("金叉回溯天數", min_value=1, max_value=10, value=3,
                                       help="過去幾天內發生金叉視為有效")
    st.divider()

    # 條件開關
    st.subheader("🔧 條件過濾")
    filter_bbw        = st.checkbox("BBW 壓縮（低於閾值）", value=True)
    filter_golden     = st.checkbox("MACD 金叉（DIF 上穿 DEA）", value=True)
    filter_below_zero = st.checkbox("DIF 在零軸以下金叉（更強勢）", value=False,
                                     help="金叉發生時 DIF < 0，起漲初期訊號更強")
    filter_hist_pos   = st.checkbox("Histogram 由負轉正", value=False,
                                     help="MACD 柱狀圖最新值 > 0")
    st.divider()

    # 股票清單
    st.subheader("📋 股票清單")
    preset = st.selectbox("快速載入", ["自訂", "半導體", "金融股", "伺服器", "全部 239 檔"])
    default_stocks = "\n".join(PRESETS[preset]) if preset != "自訂" else "\n".join(ALL_STOCKS)
    stock_input = st.text_area("股票代號（每行一個，純數字）", value=default_stocks, height=200)

# ── 主畫面 ────────────────────────────────────────
st.header("📊 BBW + MACD 金叉篩選器")
st.caption("布林通道壓縮 × DIF 上穿 DEA 複合條件篩選")

# 說明欄
with st.expander("📖 使用說明", expanded=False):
    st.markdown("""
**條件邏輯：**
- **BBW 壓縮**：布林帶寬 `(上軌 - 下軌) ÷ 中軌` 低於閾值，代表股價蓄力中
- **MACD 金叉**：DIF（快線）從下方穿越 DEA（慢線/信號線），轉為多頭訊號
- **DIF 在零軸以下**：金叉發生於 DIF < 0，代表起漲初期，訊號更強
- **Histogram 轉正**：MACD 柱狀圖由負轉正，多空力道翻轉

**最強組合**：BBW 壓縮 + 零軸下方金叉 → 蓄力後啟動，勝率最高

**FinMind 免費限制**：每天 600 次 API 請求，掃描大量股票建議分批進行
    """)

# 掃描按鈕
col_btn, col_info = st.columns([1, 3])
with col_btn:
    scan = st.button("▶ 開始掃描", type="primary", use_container_width=True)
with col_info:
    active_filters = []
    if filter_bbw: active_filters.append(f"BBW < {bb_threshold}")
    if filter_golden: active_filters.append(f"MACD 金叉（{lookback}天內）")
    if filter_below_zero: active_filters.append("DIF < 0")
    if filter_hist_pos: active_filters.append("Hist > 0")
    if active_filters:
        st.info("**啟用條件：** " + " ＋ ".join(active_filters))
    else:
        st.warning("⚠️ 所有條件已關閉，將顯示全部股票")

st.divider()

if scan:
    if not token:
        st.error("請先在左側輸入 FinMind API Token")
        st.stop()

    symbols = list({s.strip().replace(".TW", "").replace(".tw", "")
                    for s in stock_input.strip().splitlines() if s.strip()})
    if not symbols:
        st.error("請輸入至少一個股票代號")
        st.stop()

    params = {
        "bb_length": bb_length, "bb_std": bb_std, "bb_threshold": bb_threshold,
        "fast": macd_fast, "slow": macd_slow, "signal": macd_signal,
        "lookback": lookback, "fetch_days": 180,
    }

    results = []
    bbw_count = 0
    golden_count = 0
    failed = 0

    progress_bar = st.progress(0, text="準備掃描...")
    status_text  = st.empty()

    for i, sym in enumerate(symbols):
        pct = int((i + 1) / len(symbols) * 100)
        progress_bar.progress(pct, text=f"掃描中 {sym}… ({i+1}/{len(symbols)})")
        status_text.caption(f"正在處理：{sym}")

        row = analyze_stock(sym, token, params)
        if row is None:
            failed += 1
        else:
            if row["bbw_ok"]: bbw_count += 1
            if row["golden"]: golden_count += 1

            # 條件判斷
            pass_filter = True
            if filter_bbw        and not row["bbw_ok"]:  pass_filter = False
            if filter_golden     and not row["golden"]:  pass_filter = False
            if filter_below_zero and not (row["golden"] and row["dif_neg"]): pass_filter = False
            if filter_hist_pos   and not row["hist_pos"]: pass_filter = False

            if pass_filter:
                results.append(row)

        time.sleep(0.15)

    progress_bar.empty()
    status_text.empty()

    # ── 統計卡 ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("掃描總數", len(symbols))
    c2.metric("資料失敗", failed, delta=None)
    c3.metric("BBW 壓縮", bbw_count)
    c4.metric("MACD 金叉", golden_count)
    c5.metric("條件命中", len(results))

    st.divider()

    if not results:
        st.warning(f"未找到符合條件的標的（掃描 {len(symbols)} 檔，{failed} 檔失敗）")
    else:
        df_result = pd.DataFrame(results).sort_values("BBW")

        # 欄位格式化用的 helper
        def color_bbw(val):
            thr = bb_threshold
            if val < thr * 0.5:
                return "color: #f87171; font-weight: 600"
            return "color: #fbbf24; font-weight: 600"

        def color_dif(val):
            return "color: #34d399" if val >= 0 else "color: #f87171"

        def color_hist(val):
            return "color: #34d399" if val >= 0 else "color: #f87171"

        def fmt_golden(val):
            return "🟣 金叉" if val else "—"

        def fmt_bbw_tag(val):
            thr = bb_threshold
            if val < thr * 0.5:
                return "🔴 極度壓縮"
            return "🟡 壓縮中"

        # 顯示用 DataFrame
        display_df = df_result[[
            "代號", "收盤價", "BBW", "上軌", "下軌", "DIF", "DEA", "Hist", "bbw_ok", "golden"
        ]].copy()
        display_df["BB狀態"] = display_df["BBW"].apply(fmt_bbw_tag)
        display_df["MACD金叉"] = display_df["golden"].apply(fmt_golden)
        display_df = display_df.drop(columns=["bbw_ok", "golden"])

        st.dataframe(
            display_df.style
                .applymap(color_bbw, subset=["BBW"])
                .applymap(color_dif, subset=["DIF"])
                .applymap(color_hist, subset=["Hist"])
                .format({
                    "收盤價": "{:.2f}",
                    "BBW": "{:.4f}",
                    "上軌": "{:.2f}",
                    "下軌": "{:.2f}",
                    "DIF": "{:.4f}",
                    "DEA": "{:.4f}",
                    "Hist": "{:.4f}",
                }),
            use_container_width=True,
            height=min(600, 80 + len(display_df) * 36),
        )

        st.caption(f"共 {len(results)} 檔符合條件，按 BBW 由小到大排列")

        # 下載按鈕
        csv = df_result.drop(columns=["bbw_ok","golden","dif_neg","hist_pos"], errors="ignore").to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇ 下載結果 CSV",
            data=csv,
            file_name=f"bbw_macd_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
