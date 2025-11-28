import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import numpy as np

st.title("🎯 AI Smart Trader Pro — النسخة النهائية")

# --- واجهة المستخدم ---
symbol = st.selectbox("اختر الأصل:", ["AAPL","MSFT","GOOG","TSLA"])
start_date = st.date_input("تاريخ البداية")
end_date = st.date_input("تاريخ النهاية")
min_lookback = st.number_input("أيام النظر للخلف (Min)", min_value=1, value=5)
max_lookback = st.number_input("أيام النظر للخلف (Max)", min_value=min_lookback, value=40)
confidence = st.slider("حد الثقة لإشارة قوية (%)", min_value=50, max_value=95, value=70)

# زر الحصول على النتائج
if st.button("الحصول على النتائج"):
    # --- جلب البيانات ---
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        st.warning("لا توجد بيانات للأصل المحدد!")
    else:
        # --- تحويل الأعمدة إلى Series 1D ---
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # --- المؤشرات الفنية ---
        try:
            df["SMA_5"] = SMAIndicator(close, window=5).sma_indicator()
            df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
            df["EMA_10"] = EMAIndicator(close, window=10).ema_indicator()
        except Exception as e:
            st.error(f"تعذر حساب المتوسطات: {e}")

        try:
            macd = MACD(close)
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
        except Exception as e:
            st.error(f"تعذر حساب MACD: {e}")

        try:
            df["RSI_14"] = RSIIndicator(close, window=14).rsi()
        except Exception as e:
            st.error(f"تعذر حساب RSI: {e}")

        try:
            df["Volume_SMA"] = SMAIndicator(volume, window=20).sma_indicator()
            df["Volume_Ratio"] = volume / df["Volume_SMA"].replace(0,np.nan)
        except Exception as e:
            st.error(f"تعذر حساب Volume Ratio: {e}")

        # --- إشارات التداول ---
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            signals = []
            for i in range(len(df)):
                if not pd.isna(df["MACD"].iloc[i]) and not pd.isna(df["MACD_signal"].iloc[i]):
                    if df["MACD"].iloc[i] > df["MACD_signal"].iloc[i]:
                        signals.append("شراء")
                    else:
                        signals.append("بيع")
                else:
                    signals.append("")
            df["Signal"] = signals

        # --- عرض النتائج ---
        st.subheader("📈 بيانات الأسعار والمؤشرات")
        st.line_chart(df[["Close","SMA_5","SMA_20","EMA_10"]].tail(150))

        if "Signal" in df.columns:
            st.subheader("🎯 إشارات التداول")
            st.dataframe(df[["Close","MACD","MACD_signal","RSI_14","Signal"]].tail(50))