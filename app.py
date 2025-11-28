import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

# -------------------------------
# واجهة التطبيق
# -------------------------------
st.title("🎯 AI Smart Trader Pro — النسخة النهائية")
st.write("تحليل الأسهم باستخدام الذكاء الاصطناعي")

# اختيار الأصل وتواريخ التحليل
symbol = st.selectbox("اختر الأصل:", ["AAPL", "GOOGL", "MSFT"])
start_date = st.date_input("تاريخ البداية")
end_date = st.date_input("تاريخ النهاية")
min_lookback = st.number_input("أيام النظر للخلف (Min)", min_value=1, value=5)
max_lookback = st.number_input("أيام النظر للخلف (Max)", min_value=min_lookback, value=40)
confidence = st.slider("حد الثقة لإشارة قوية (%)", 0, 100, (50, 95))

# زر الحصول على النتائج
if st.button("الحصول على النتائج"):
    # -------------------------------
    # تحميل البيانات
    # -------------------------------
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        st.error("لا توجد بيانات لهذا المدى الزمني.")
    else:
        # تأكد من أن الأعمدة 1D
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        
        # -------------------------------
        # حساب المؤشرات الفنية
        # -------------------------------
        try:
            df["SMA_5"] = SMAIndicator(close, window=5).sma_indicator()
        except Exception as e:
            st.warning(f"تعذر حساب SMA_5: {e}")

        try:
            df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
        except Exception as e:
            st.warning(f"تعذر حساب SMA_20: {e}")

        try:
            df["EMA_10"] = EMAIndicator(close, window=10).ema_indicator()
        except Exception as e:
            st.warning(f"تعذر حساب EMA_10: {e}")

        try:
            macd = MACD(close)
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
        except Exception as e:
            st.warning(f"تعذر حساب MACD: {e}")

        try:
            df["RSI"] = RSIIndicator(close).rsi()
        except Exception as e:
            st.warning(f"تعذر حساب RSI: {e}")

        try:
            df["Volume_SMA"] = SMAIndicator(volume, window=20).sma_indicator()
            df["Volume_Ratio"] = volume / df["Volume_SMA"].replace(0, np.nan)
        except Exception as e:
            st.warning(f"تعذر حساب Volume Ratio: {e}")

        # -------------------------------
        # عرض النتائج
        # -------------------------------
        st.subheader("📈 بيانات الأسعار والمؤشرات")
        columns_to_plot = ["Close","SMA_5","SMA_20","EMA_10"]
        existing_columns = [col for col in columns_to_plot if col in df.columns]
        
        if existing_columns:
            st.line_chart(df[existing_columns].tail(150))
        else:
            st.warning("لا توجد أعمدة صالحة للرسم البياني.")
        
        st.subheader("💡 بيانات أولية")
        st.dataframe(df.tail(10))