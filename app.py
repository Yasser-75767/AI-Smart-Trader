import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD

# --- واجهة التطبيق الأصلية ---
st.title("🎯 AI Smart Trader Pro — النسخة النهائية")
st.subheader("تحليل الأسهم باستخدام الذكاء الاصطناعي")

# --- إدخالات المستخدم ---
symbol = st.selectbox("اختر الأصل:", ["AAPL", "TSLA", "GOOGL", "MSFT"])
start_date = st.date_input("تاريخ البداية")
end_date = st.date_input("تاريخ النهاية")
lookback = st.slider("أيام النظر للخلف:", min_value=5, max_value=40, value=20)
confidence = st.slider("حد الثقة لإشارة قوية (%):", min_value=50, max_value=95, value=70)

# --- جلب البيانات ---
df = yf.download(symbol, start=start_date, end=end_date)
df.reset_index(inplace=True)

# --- التأكد من عدم وجود قيم مفقودة ---
df.fillna(method='ffill', inplace=True)

# --- حساب المتوسطات المتحركة ---
df["SMA_5"] = SMAIndicator(df["Close"], window=5).sma_indicator()
df["SMA_20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()

# --- حساب MACD ---
macd_indicator = MACD(df["Close"])
df["MACD"] = macd_indicator.macd()

# --- حساب نسبة الحجم ---
df["Volume_SMA"] = df["Volume"].rolling(window=20).mean().replace(0, np.nan)
df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]

# --- عرض النتائج ---
st.subheader("📈 السعر + المتوسطات المتحركة")
st.line_chart(df[["Close", "SMA_20", "SMA_50"]].tail(150))

st.subheader("💹 إشارات التداول")
if df["MACD"].iloc[-1] > 0 and df["Close"].iloc[-1] > df["SMA_20"].iloc[-1]:
    st.success("إشارة شراء قوية ✅")
else:
    st.warning("إشارة للبيع ⚠️")

st.subheader("🔢 بيانات الحجم")
st.line_chart(df[["Volume", "Volume_SMA", "Volume_Ratio"]].tail(150))