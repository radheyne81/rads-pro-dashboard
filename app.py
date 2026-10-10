import streamlit as st
import yfinance as yf
import pandas as pd
import concurrent.futures
import warnings

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Rads Pro Dashboard", layout="wide", page_icon="⚡")

# --- CUSTOM CSS STYLING & BEAUTIFICATION ---
st.markdown("""
    <style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #00D2FF; margin-bottom: 2px; }
    .sub-header { font-size: 1.05rem; color: #A0B2C6; margin-bottom: 22px; font-weight: 400; }
    .stButton>button { 
        background: linear-gradient(90deg, #00D2FF 0%, #0084FF 100%);
        color: #ffffff; 
        font-weight: 700; 
        font-size: 1rem;
        border-radius: 8px; 
        border: none;
        padding: 0.6rem 1rem;
        width: 100%; 
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Rads Pro: Multi-Timeframe Terminal & Strategy Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated screening across Indian Indices, Sectors, and Stocks with technical rationality and rule-based action protocols.</div>', unsafe_allow_html=True)

# --- ASSET UNIVERSE ---
ASSETS = {
    "Indices & Sectors": {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "NIFTY IT": "^CNXIT",
        "NIFTY AUTO": "^CNXAUTO",
        "NIFTY METAL": "^CNXMETAL",
        "NIFTY PHARMA": "^CNXPHARMA",
        "NIFTY FMCG": "^CNXFMCG",
    },
    "Nifty Stocks": {
        "RELIANCE": "RELIANCE.NS",
        "HDFC BANK": "HDFCBANK.NS",
        "ICICI BANK": "ICICIBANK.NS",
        "TCS": "TCS.NS",
        "INFOSYS": "INFY.NS",
        "TATA MOTORS": "TATAMOTORS.NS",
        "M&M": "M&M.NS",
        "ITC": "ITC.NS",
        "L&T": "LT.NS",
        "STATE BANK OF INDIA": "SBIN.NS",
        "BAJAJ FINANCE": "BAJFINANCE.NS",
        "DIXON TECH": "DIXON.NS",
        "PFC": "PFC.NS",
        "REC LTD": "RECLTD.NS",
        "ZOMATO": "ZOMATO.NS"
    }
}

# --- NATIVE MATH FUNCTIONS (Replaces pandas-ta) ---
def get_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_bb(series, period=20, std=2):
    sma = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    width = (upper - lower) / sma
    return upper, width

# --- ALGORITHM & DATA FETCHING ENGINE ---
def analyze_asset(name, ticker):
    try:
        df_daily = yf.download(ticker, period="2y", progress=False)
        if df_daily.empty: return None
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.droplevel(1)
        df_daily.dropna(inplace=True)

        df_15m = yf.download(ticker, period="60d", interval="15m", progress=False)
        if df_15m.empty: return None
        if isinstance(df_15m.columns, pd.MultiIndex):
            df_15m.columns = df_15m.columns.droplevel(1)
        df_15m.dropna(inplace=True)

        df_w = df_daily.resample('W-FRI').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        df_m = df_daily.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        df_1h = df_15m.resample('1h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

        # Compute RSI
        df_m['RSI'] = get_rsi(df_m['Close'], 14)
        df_w['RSI'] = get_rsi(df_w['Close'], 14)
        df_daily['RSI'] = get_rsi(df_daily['Close'], 14)
        df_1h['RSI'] = get_rsi(df_1h['Close'], 14)
        df_15m['RSI'] = get_rsi(df_15m['Close'], 14)

        # Compute BB
        df_daily['BB_Upper'], df_daily['BB_Width'] = get_bb(df_daily['Close'], 20, 2)

        m_rsi = round(df_m['RSI'].iloc[-1], 1) if not df_m.empty and not pd.isna(df_m['RSI'].iloc[-1]) else None
        w_rsi = round(df_w['RSI'].iloc[-1], 1) if not df_w.empty and not pd.isna(df_w['RSI'].iloc[-1]) else None
        d_rsi = round(df_daily['RSI'].iloc[-1], 1)
        h_rsi = round(df_1h['RSI'].iloc[-1], 1) if not df_1h.empty and not pd.isna(df_1h['RSI'].iloc[-1]) else None
        m15_rsi = round(df_15m['RSI'].iloc[-1], 1) if not df_15m.empty and not pd.isna(df_15m['RSI'].iloc[-1]) else None

        last_d = df_daily.iloc[-1]
        prev_d = df_daily.iloc[-2]
        o, h, l, c = last_d['Open'], last_d['High'], last_d['Low'], last_d['Close']
        prev_c = prev_d['Close']

        body = abs(c - o)
        lower_wick = (o - l) if c > o else (c - l)
        upper_wick = (h - c) if c > o else (h - o)

        pa_signal = "🟩 Green" if c > o else "🟥 Red"
        if l > prev_c * 1.004: pa_signal += " | 🚀 Gap Up"
        elif h < prev_c * 0.996: pa_signal += " | 🩸 Gap Down"
        if lower_wick > (2 * body) and upper_wick < body: pa_signal += " | 🔨 Bull Hammer"
        if upper_wick > (2 * body) and lower_wick < body: pa_signal += " | 🌠 Shoot Star"

        bb_upper = last_d['BB_Upper']
        bb_width = last_d['BB_Width']

        setup = "Sideways / Neutral"
        logic = "Daily RSI inside 40-60 range without multi-timeframe alignment"
        action = "Wait / Sideways (Deploy non-directional option selling if applicable)"

        if m_rsi and w_rsi and m_rsi >= 60 and w_rsi >= 60 and (38 <= d_rsi <= 45):
            setup = "🔥 Macro GFS Setup"
            logic = f"Monthly RSI ({m_rsi}) & Weekly RSI ({w_rsi}) ≥ 60 confirm macro uptrend; Daily RSI ({d_rsi}) testing 40-support"
            action = "Buy on confirmed Green Day candle; Stop Loss below swing low"
        elif h_rsi and m15_rsi and d_rsi >= 60 and h_rsi >= 60 and (38 <= m15_rsi <= 45):
            setup = "⚡ Micro GFS (Intraday)"
            logic = f"Daily RSI ({d_rsi}) & Hourly RSI ({h_rsi}) ≥ 60; 15M RSI ({m15_rsi}) pulling back to 40 support"
            action = "Buy on 15m green reversal; Stop Loss below 15m trigger candle low"
        elif bb_width < 0.05 and d_rsi > 60 and c > bb_upper:
            setup = "🚀 BB Squeeze Breakout"
            logic = f"BB bandwidth compressed (< 5%); Close broke upper band with RSI ({d_rsi}) > 60"
            action = "Enter momentum breakout long; Trail using swing lows"
        elif (38 <= d_rsi <= 42) and (w_rsi is not None and w_rsi < 60):
            setup = "🛡️ Bullish Range Shift"
            logic = f"Daily RSI held 40 support ({d_rsi}) rejecting lower bearish band (<40)"
            action = "Accumulate on pullbacks; Confirm with Higher Low"
        elif m_rsi and w_rsi and d_rsi >= 60 and w_rsi >= 60 and m_rsi >= 60:
            setup = "🟢 Super Bullish (>60)"
            logic = f"All active macro timeframes sustaining above 60 momentum threshold"
            action = "Maintain long positions; Avoid shorting; Trail Stop Loss"
        elif w_rsi and d_rsi <= 40 and w_rsi <= 40:
            setup = "🔴 Bearish Zone (<40)"
            logic = f"Weekly ({w_rsi}) and Daily ({d_rsi}) sustaining in <40 bear momentum zone"
            action = "Avoid long trades; Sell on rallies near 60 RSI"

        return {
            "Asset": name,
            "Price (₹)": round(c, 2),
            "Setup": setup,
            "Setup Logic": logic,
            "Suggested Action": action,
            "Price Action": pa_signal,
            "Monthly": m_rsi,
            "Weekly": w_rsi,
            "Daily": d_rsi,
            "Hourly (1H)": h_rsi,
            "15 Min": m15_rsi
        }
    except Exception as e:
        return None

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### ⚙️ Scanner Universe")
scan_category = st.sidebar.radio("Select Universe:", ["All Assets", "Indices & Sectors Only", "Nifty Stocks Only"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Flexi-Select Timeframes")
selected_tfs = st.sidebar.multiselect(
    "Active Timeframes:",
    ["Monthly", "Weekly", "Daily", "Hourly (1H)", "15 Min"],
    default=["Monthly", "Weekly", "Daily", "Hourly (1H)", "15 Min"]
)

# --- EXECUTION ---
if st.sidebar.button("⚡ Run Rads Pro Scan"):
    with st.spinner("Fetching OHLCV data from NSE & evaluating rule parameters..."):
        
        target_assets = {}
        if scan_category in ["All Assets", "Indices & Sectors Only"]: 
            target_assets.update(ASSETS["Indices & Sectors"])
        if scan_category in ["All Assets", "Nifty Stocks Only"]: 
            target_assets.update(ASSETS["Nifty Stocks"])

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(analyze_asset, name, ticker): name for name, ticker in target_assets.items()}
            for future in concurrent.futures.as_completed(future_to_ticker):
                res = future.result()
                if res: results.append(res)

        if results:
            df = pd.DataFrame(results)
            core_cols = ["Asset", "Price (₹)", "Setup", "Setup Logic", "Suggested Action", "Price Action"]
            df = df[core_cols + selected_tfs]

            def color_rsi(val):
                if pd.isna(val): return ''
                if val >= 60: return 'background-color: #004d00; color: #a3f7a3; font-weight: 600;'
                elif val <= 40: return 'background-color: #5c0e0e; color: #fca5a5; font-weight: 600;'
                else: return 'background-color: #2b2b2b; color: #d1d5db;'

            def color_setup(val):
                if 'Macro GFS' in val: return 'background-color: #007744; color: #ffffff; font-weight: 700;'
                if 'Micro GFS' in val: return 'background-color: #005f73; color: #ffffff; font-weight: 700;'
                if 'Breakout' in val: return 'background-color: #1e3a8a; color: #ffffff; font-weight: 700;'
                if 'Range Shift' in val: return 'background-color: #854d0e; color: #ffffff; font-weight: 700;'
                if 'Super Bullish' in val: return 'background-color: #064e3b; color: #34d399; font-weight: 600;'
                if 'Bearish Zone' in val: return 'background-color: #7f1d1d; color: #f87171; font-weight: 600;'
                return ''

            st.success("✅ Analysis complete.")

            st.markdown("### 🔎 Quick Filter by Setup")
            filter_setup = st.selectbox("Isolate specific setups:", ["Show All"] + sorted(list(df['Setup'].unique())))

            if filter_setup != "Show All":
                df_filtered = df[df['Setup'] == filter_setup]
            else:
                df_filtered = df

            styled_df = df_filtered.style.map(color_rsi, subset=selected_tfs) \
                                         .map(color_setup, subset=['Setup'])
            
            st.dataframe(styled_df, use_container_width=True, height=720)
