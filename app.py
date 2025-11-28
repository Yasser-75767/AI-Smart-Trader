import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.title("🎯 AI Smart Trader Pro — النسخة الكاملة النهائية")

# ---- واجهة المستخدم ----
symbol = st.selectbox("اختر الأصل:", ["AAPL", "MSFT", "GOOG", "TSLA"])
start_date = st.date_input("تاريخ البداية")
end_date = st.date_input("تاريخ النهاية")
lookback_min = st.number_input("أيام النظر للخلف (Min)", min_value=5, max_value=40, value=5)
lookback_max = st.number_input("أيام النظر للخلف (Max)", min_value=5, max_value=40, value=20)
confidence = st.slider("حد الثقة لإشارة قوية (%)", min_value=50, max_value=95, value=70)

# ---- زر الحصول على النتائج ----
if st.button("الحصول على النتائج"):
    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty or len(df) < 5:
        st.error("اختر فترة زمنية أطول، البيانات غير كافية لحساب المؤشرات.")
    else:
        # ----- حساب المؤشرات -----
        try:
            df["SMA_5"] = SMAIndicator(df["Close"], window=5).sma_indicator()
            df["SMA_20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
            df["EMA_10"] = EMAIndicator(df["Close"], window=10).ema_indicator()
        except Exception as e:
            st.warning(f"تعذر حساب المتوسطات: {e}")

        try:
            macd = MACD(df["Close"])
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
        except Exception as e:
            st.warning(f"تعذر حساب MACD: {e}")

        try:
            df["RSI_14"] = RSIIndicator(df["Close"], window=14).rsi()
        except Exception as e:
            st.warning(f"تعذر حساب RSI: {e}")

        try:
            df["Volume_SMA"] = SMAIndicator(df["Volume"], window=20).sma_indicator()
            df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"].replace(0, np.nan)
        except Exception as e:
            st.warning(f"تعذر حساب Volume Ratio: {e}")

        # ----- إشارات تداول مبسطة -----
        signals = []
        for i in range(len(df)):
            signal = ""
            if not pd.isna(df["MACD"].iloc[i]) and not pd.isna(df["MACD_signal"].iloc[i]):
                if df["MACD"].iloc[i] > df["MACD_signal"].iloc[i]:
                    signal = f"شراء (ثقة {confidence}%)"
                elif df["MACD"].iloc[i] < df["MACD_signal"].iloc[i]:
                    signal = f"بيع (ثقة {confidence}%)"
            signals.append(signal)
        df["Signal"] = signals

        # ----- عرض النتائج -----
        st.subheader("البيانات الأخيرة")
        st.dataframe(df.tail(10))

        st.subheader("رسم السعر والمتوسطات المتحركة")
        st.line_chart(df[["Close", "SMA_5", "SMA_20", "EMA_10"]].dropna())

        st.subheader("إشارات التداول")
        st.dataframe(df[["Close", "MACD", "MACD_signal", "RSI_14", "Volume_Ratio", "Signal"]].tail(10))

        st.success("تم الحساب بنجاح ✅")