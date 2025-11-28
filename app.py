# AI Smart Trader Pro — النسخة النهائية المُعدّلة
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="AI Smart Trader Pro", layout="wide")

st.title("🎯 AI Smart Trader Pro — النسخة النهائية")

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
volume = df["Volume"]

# حساب المتوسطات المتحركة
df["SMA_5"] = close.rolling(window=5).mean()
df["SMA_20"] = close.rolling(window=20).mean()
df["SMA_50"] = close.rolling(window=50).mean()

# حساب MACD بطريقة آمنة 1D
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

# حساب Volume Ratio بطريقة آمنة
df["Volume_SMA"] = volume.rolling(window=20).mean().replace(0, np.nan)
df["Volume_Ratio"] = volume / df["Volume_SMA"]

# حساب Gap
df["Gap"] = (open_ - close.shift(1)) / close.shift(1)

# عرض البيانات والرسوم
st.subheader("📈 السعر + المتوسطات المتحركة")
st.line_chart(df[["Close", "SMA_20", "SMA_50"]].tail(150))

st.subheader("MACD")
st.line_chart(df[["MACD", "MACD_Signal"]].tail(150))

st.subheader("Volume Ratio")
st.line_chart(df["Volume_Ratio"].tail(150))

# إشارات تداول بسيطة
st.subheader("🎯 إشارة التداول الحالية")
latest_macd = df["MACD"].iloc[-1]
latest_signal = df["MACD_Signal"].iloc[-1]

if latest_macd > latest_signal:
    st.success("شراء ↑ (MACD فوق الإشارة)")
else:
    st.error("بيع ↓ (MACD تحت الإشارة)")

st.info("💡 هذا التطبيق تعليمي + عملي، يمكن استخدامه للتداول الفعلي (احذر المخاطر).")