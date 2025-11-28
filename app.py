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
    # جلب البيانات
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        st.error("لا توجد بيانات متاحة لهذا الرمز.")
    else:
        # تأكد أن الأعمدة هي 1D
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # حساب المتوسطات والمؤشرات مع حماية من الأخطاء 1D
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

        # عرض بيانات الأسعار والمتوسطات
        columns_to_plot = ["Close", "SMA_5", "SMA_20", "EMA_10"]
        existing_columns = [col for col in columns_to_plot if col in df.columns]
        if existing_columns:
            st.subheader("📈 بيانات الأسعار والمتوسطات")
            st.line_chart(df[existing_columns].tail(150))
        else:
            st.warning("لا توجد أعمدة صالحة للرسم البياني.")

        # حساب إشارات التداول
        df["Signal"] = np.nan
        for i in range(1, len(df)):
            if "MACD" in df.columns and "MACD_signal" in df.columns:
                if not pd.isna(df["MACD"].iloc[i]) and not pd.isna(df["MACD_signal"].iloc[i]):
                    if df["MACD"].iloc[i] > df["MACD_signal"].iloc[i]:
                        df["Signal"].iloc[i] = "شراء"
                    elif df["MACD"].iloc[i] < df["MACD_signal"].iloc[i]:
                        df["Signal"].iloc[i] = "بيع"

        st.subheader("🎯 إشارات التداول")
        st.dataframe(df[["Close","Signal"]].tail(20))