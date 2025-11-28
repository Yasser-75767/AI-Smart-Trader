import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.title("🎯 AI Smart Trader Pro — النسخة النهائية مع إشارات التداول")

# إعدادات المستخدم
symbol = st.selectbox("اختر الأصل:", ["AAPL", "GOOGL", "MSFT", "TSLA"])
start_date = st.date_input("تاريخ البداية")
end_date = st.date_input("تاريخ النهاية")
min_lookback = st.number_input("أيام النظر للخلف (Min)", min_value=1, value=5)
max_lookback = st.number_input("أيام النظر للخلف (Max)", min_value=min_lookback, value=40)
confidence_min = st.slider("حد الثقة لإشارة قوية (%)", 0, 100, 50, 1)
confidence_max = st.slider("حد الثقة لإشارة قوية (%)", 0, 100, 95, 1)

# زر الحصول على النتائج
if st.button("الحصول على النتائج"):
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        st.error("لا توجد بيانات متاحة لهذا الرمز.")
    else:
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # حساب المتوسطات والمؤشرات مع حماية 1D
        indicators = {}
        try:
            indicators["SMA_5"] = SMAIndicator(close, window=5).sma_indicator().squeeze()
        except:
            pass
        try:
            indicators["SMA_20"] = SMAIndicator(close, window=20).sma_indicator().squeeze()
        except:
            pass
        try:
            indicators["EMA_10"] = EMAIndicator(close, window=10).ema_indicator().squeeze()
        except:
            pass
        try:
            macd = MACD(close)
            indicators["MACD"] = macd.macd().squeeze()
            indicators["MACD_signal"] = macd.macd_signal().squeeze()
        except:
            pass
        try:
            indicators["RSI"] = RSIIndicator(close).rsi().squeeze()
        except:
            pass
        try:
            indicators["Volume_SMA"] = SMAIndicator(volume, window=20).sma_indicator().squeeze()
            indicators["Volume_Ratio"] = volume / indicators["Volume_SMA"].replace(0, np.nan)
        except:
            pass

        # دمج المؤشرات في DataFrame
        for name, series in indicators.items():
            df[name] = series

        # رسم البيانات المتوفرة فقط
        columns_to_plot = ["Close","SMA_5","SMA_20","EMA_10"]
        existing_columns = [col for col in columns_to_plot if col in df.columns]
        if existing_columns:
            st.subheader("📈 بيانات الأسعار والمتوسطات")
            st.line_chart(df[existing_columns].tail(150))
        else:
            st.warning("لا توجد أعمدة صالحة للرسم البياني.")

        # إشارات التداول بناءً على MACD
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            df["Signal"] = np.where(df["MACD"] > df["MACD_signal"], "شراء",
                             np.where(df["MACD"] < df["MACD_signal"], "بيع", np.nan))
            st.subheader("🎯 إشارات التداول")
            st.dataframe(df[["Close","Signal"]].tail(20))
        else:
            st.warning("MACD غير متوفر، لا يمكن حساب الإشارات.")