# app.py — AI Smart Trader Pro (نسخة مصححة بالكامل)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import ta
import datetime
import warnings
import random

warnings.filterwarnings("ignore")

# ================= إعداد الصفحة =================
st.set_page_config(page_title="AI Smart Trader Pro — النسخة المصححة", layout="wide")

# ================= القوائم =================
stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN","TSLA", "META", "JPM", "JNJ", "V","WMT", "PG", "DIS", "NFLX", "ADBE"]
forex_pairs = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X","USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"]
crypto = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LTC-USD","XRP-USD", "DOGE-USD", "SOL-USD", "AVAX-USD", "MATIC-USD"]
all_symbols = stocks + forex_pairs + crypto

# ================= الشريط الجانبي =================
st.sidebar.header("⚙️ الإعدادات المتقدمة")
symbol = st.sidebar.selectbox("اختر الأصل:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
model_type = st.sidebar.selectbox("نموذج التعلم الآلي:", ["XGBoost", "Random Forest", "Gradient Boosting", "Ensemble"])
lookback_days = st.sidebar.slider("أيام النظر للخلف:", 5, 40, 20)
test_size = st.sidebar.slider("حجم بيانات الاختبار (%):", 10, 40, 20)
confidence_threshold = st.sidebar.slider("حد الثقة لإشارة قوية (%):", 50, 95, 75)

if "run" not in st.session_state:
    st.session_state["run"] = False

# ================= دوال المساعدة =================
def fetch_market_sentiment():
    return {"bullish": random.uniform(0.4, 0.7),"bearish": random.uniform(0.2, 0.6),"neutral": random.uniform(0.1, 0.3)}

def load_enhanced_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty or len(df) < 100:
        return pd.DataFrame()

    # التأكد من أن كل الأعمدة Series أحادية البعد
    close = df["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    close = close.dropna()

    high = df["High"].dropna()
    low = df["Low"].dropna()
    volume = df["Volume"].dropna()

    # مؤشرات فنية
    df["SMA_5"] = ta.trend.sma_indicator(close, window=5)
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    df["EMA_10"] = ta.trend.ema_indicator(close, window=10)
    df["EMA_50"] = ta.trend.ema_indicator(close, window=50)
    df["RSI_14"] = ta.momentum.rsi(close, window=14)
    df["MACD"] = ta.trend.macd(close)
    df["MACD_Signal"] = ta.trend.macd_signal(close)
    df["MACD_Hist"] = ta.trend.macd_diff(close)
    df["BB_Upper"] = ta.volatility.bollinger_hband(close)
    df["BB_Lower"] = ta.volatility.bollinger_lband(close)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["Stoch_K"] = ta.momentum.stoch(high, low, close)
    df["Stoch_D"] = ta.momentum.stoch_signal(high, low, close)
    df["Williams_R"] = ta.momentum.williams_r(high, low, close)
    df["CCI"] = ta.trend.cci(high, low, close)
    df["ADX"] = ta.trend.adx(high, low, close)
    df["OBV"] = ta.volume.on_balance_volume(close, volume)
    df["ATR"] = ta.volatility.average_true_range(high, low, close)
    df["Returns"] = close.pct_change()
    df["Log_Returns"] = np.log(close / close.shift(1))
    df["Price_Range"] = (high - low) / close
    df["Gap"] = (df["Open"] - close.shift(1)) / close.shift(1)
    df["Volatility_5D"] = df["Returns"].rolling(5).std()
    df["Volatility_21D"] = df["Returns"].rolling(21).std()
    df["Momentum_5D"] = close / close.shift(5) - 1
    df["Momentum_21D"] = close / close.shift(21) - 1
    df["Volume_MA20"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / df["Volume_MA20"].replace(0, np.nan)
    df["Day_of_Week"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter

    return df.dropna()

def create_advanced_target(df, lookahead=1):
    future_return = df["Close"].shift(-lookahead) / df["Close"] - 1
    df["Binary_Target"] = (future_return > 0).astype(int)
    return df.dropna()

# ================= واجهة التطبيق =================
st.title("🎯 AI Smart Trader Pro — النسخة المصححة")
st.markdown("نظام متقدم للتحليل والتنبؤ المالي باستخدام الذكاء الاصطناعي.")
st.info("💡 مهم: الأسواق خطيرة، كل التداول على مسؤوليتك.")

run_button = st.button("🚀 بدء التحليل المتقدم")
if run_button:
    st.session_state["run"] = True

if st.session_state.get("run", False):
    with st.spinner('جاري تحميل البيانات...'):
        df = load_enhanced_data(symbol, start_date, end_date)
    if df.empty:
        st.error("❌ البيانات غير كافية لهذا الأصل.")
    else:
        df = create_advanced_target(df)
        st.success("✅ البيانات جاهزة والتحليل مستعد للعمل!")
        st.dataframe(df.tail(10))