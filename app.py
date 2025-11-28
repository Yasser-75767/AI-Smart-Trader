# app_safe.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from PIL import Image
import cv2
import random
import datetime
from streamlit_autorefresh import st_autorefresh

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader Live 💜", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Price_Range", "Price_Change", "MA_5", "Volume_MA"
]

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر سهم أو زوج الفوركس:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
uploaded_file = st.sidebar.file_uploader("ارفع صورة الشموع/المنحنيات للتحليل", type=["png","jpg","jpeg"])
auto_refresh = st.sidebar.number_input("تحديث تلقائي بالثواني:", min_value=1, max_value=3600, value=10, step=1)

# ===== تحقق التواريخ =====
if start_date >= end_date or start_date > datetime.date.today():
    st.sidebar.error("⚠ تواريخ غير صحيحة")
    st.stop()

# ===== التحديث التلقائي =====
if auto_refresh:
    st_autorefresh(interval=auto_refresh*1000, limit=None, key="autorefresh")

# ===== تحميل البيانات مع بدائل =====
def load_data_safe(symbol, start, end):
    candidates = [symbol] + [s for s in all_symbols if s != symbol]
    for sym in candidates:
        try:
            df = yf.download(sym, start=start, end=end, progress=False)
            base_cols = ["Open","High","Low","Close","Volume"]
            if df.empty or not all(c in df.columns for c in base_cols):
                continue
            df = df[base_cols].dropna()
            if len(df) < 5:
                continue
            if sym != symbol:
                st.info(f"ℹ تم استخدام {sym} بدل {symbol} لعدم توفر بيانات كافية")
            return df, sym
        except:
            continue
    return pd.DataFrame(), symbol

# ===== تجهيز الميزات =====
def prepare_features_safe(df, with_target=True):
    try:
        df = df.copy()
        base_cols = ["Open","High","Low","Close","Volume"]
        for col in base_cols:
            if col not in df.columns:
                df[col] = 0.0
        df["Price_Range"] = df["High"] - df["Low"]
        df["Price_Change"] = df["Close"] - df["Open"]
        df["MA_5"] = df["Close"].rolling(5).mean().fillna(0)
        df["Volume_MA"] = df["Volume"].rolling(5).mean().fillna(0)
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
        if with_target:
            df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
            df = df.dropna(subset=["Target"]) if "Target" in df.columns else df
            X = df[FEATURE_COLS]
            y = df["Target"].astype(int) if "Target" in df.columns else pd.Series(np.zeros(len(df)))
            return X, y, df
        else