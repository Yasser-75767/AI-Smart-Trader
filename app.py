# AI Smart Trader Pro — النسخة الكاملة النهائية
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="AI Smart Trader Pro", layout="wide")
st.title("🎯 AI Smart Trader Pro — النسخة الكاملة النهائية")

# ⚙️ الإعدادات المتقدمة
symbol = st.text_input("اختر الأصل:", value="AAPL")
start_date = st.date_input("تاريخ البداية:", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("تاريخ النهاية:", value=pd.to_datetime(datetime.today()))
lookback = st.slider("أيام النظر للخلف:", 5, 40, 20)
confidence = st.slider("حد الثقة لإشارة قوية (%):", 50, 95, 70)

# تحميل البيانات
@st.cache_data
def load_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df.dropna()
    return df

df = load_data(symbol, start_date, end_date)
close = df["Close"]
open_ = df["Open"]
high = df["High"]
low = df["Low"]
volume = df["Volume"]

# --- المؤشرات الفنية ---
# المتوسطات المتحركة
df["SMA_5"] = close.rolling(window=5).mean()
df["SMA_20"] = close.rolling(window=20).mean()
df["SMA_50"] = close.rolling(window=50).mean()

df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

# MACD
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

# RSI
delta = close.diff()
gain = delta.clip(lower=0)
loss = -1*delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# Bollinger Bands
df["BB_Mid"] = close.rolling(window=20).mean()
df["BB_Std"] = close.rolling(window=20).std()
df["BB_Upper"] = df["BB_Mid"] + 2*df["BB_Std"]
df["BB_Lower"] = df["BB_Mid"] - 2*df["BB_Std"]

# Volume Ratio
df["Volume_SMA"] = volume.rolling(window=20).mean().replace(0, np.nan)
df["Volume_Ratio"] = volume / df["Volume_SMA"]

# Gap
df["Gap"] = (open_ - close.shift(1)) / close.shift(1)

# --- الرسوم البيانية ---
st.subheader("📈 السعر + المتوسطات المتحركة")
st.line_chart(df[["Close", "SMA_20", "SMA_50"]].tail(150))

st.subheader("MACD")
st.line_chart(df[["MACD", "MACD_Signal"]].tail(150))

st.subheader("RSI")
st.line_chart(df["RSI"].tail(150))

st.subheader("Bollinger Bands")
st.line_chart(df[["Close", "BB_Upper", "BB_Lower"]].tail(150))

st.subheader("Volume Ratio")
st.line_chart(df["Volume_Ratio"].tail(150))

# --- إشارات تداول ذكية ---
st.subheader("🎯 إشارة التداول الحالية")
latest_macd = df["MACD"].iloc[-1]
latest_signal = df["MACD_Signal"].iloc[-1]
latest_rsi = df["RSI"].iloc[-1]
latest_price = close.iloc[-1]
upper_bb = df["BB_Upper"].iloc[-1]
lower_bb = df["BB_Lower"].iloc[-1]

signal = ""
if latest_macd > latest_signal and latest_rsi < 70 and latest_price < upper_bb:
    signal = "شراء ↑"
elif latest_macd < latest_signal and latest_rsi > 30 and latest_price > lower_bb:
    signal = "بيع ↓"
else:
    signal = "تثبيت ⚖️"

st.info(f"الإشارة الحالية: {signal}")
st.info("💡 هذا التطبيق تعليمي + عملي، يمكن استخدامه للتداول الفعلي (احذر المخاطر).")