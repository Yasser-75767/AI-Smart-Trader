import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.set_page_config(page_title="🎯 AI Smart Trader Pro", layout="wide")
st.title("🎯 AI Smart Trader Pro — النسخة النهائية مع إشارات التداول")
st.markdown("تحليل الأسهم باستخدام الذكاء الاصطناعي")

# -------------------------------
# إعدادات المستخدم
# -------------------------------
symbol = st.selectbox("اختر الأصل:", ["AAPL", "GOOGL", "MSFT"])
start_date = st.date_input("تاريخ البداية")
end_date = st.date_input("تاريخ النهاية")
min_lookback = st.number_input("أيام النظر للخلف (Min)", min_value=1, value=5)
max_lookback = st.number_input("أيام النظر للخلف (Max)", min_value=min_lookback, value=40)
confidence_min = st.slider("حد الثقة لإشارة قوية (%)", min_value=0, max_value=100, value=50)
confidence_max = st.slider("حد الثقة لإشارة قوية (%)", min_value=confidence_min, max_value=100, value=95)

# -------------------------------
# زر الحساب
# -------------------------------
if st.button("الحصول على النتائج"):
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        st.error("لا توجد بيانات لهذا المدى الزمني.")
    else:
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # -------------------------------
        # حساب المتوسطات والمؤشرات
        # -------------------------------
        try:
            df["SMA_5"] = SMAIndicator(close, window=5).sma_indicator()
            df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
            df["EMA_10"] = EMAIndicator(close, window=10).ema_indicator()
        except Exception as e:
            st.warning(f"تعذر حساب المتوسطات: {e}")

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
        # إشارات التداول
        # -------------------------------
        signals = []
        if "MACD" in df.columns and "MACD_signal" in df.columns and "RSI" in df.columns:
            for i in range(len(df)):
                if not pd.isna(df["MACD"].iloc[i]) and not pd.isna(df["MACD_signal"].iloc[i]) and not pd.isna(df["RSI"].iloc[i]):
                    if df["MACD"].iloc[i] > df["MACD_signal"].iloc[i] and df["RSI"].iloc[i] < 30:
                        signals.append("Buy")
                    elif df["MACD"].iloc[i] < df["MACD_signal"].iloc[i] and df["RSI"].iloc[i] > 70:
                        signals.append("Sell")
                    else:
                        signals.append("Hold")
                else:
                    signals.append("Hold")
            df["Signal"] = signals
        else:
            st.warning("لا يمكن حساب إشارات التداول بسبب نقص البيانات.")

        # -------------------------------
        # الرسم البياني للأسعار والمتوسطات
        # -------------------------------
        columns_to_plot = ["Close", "SMA_5", "SMA_20", "EMA_10"]
        existing_columns = [col for col in columns_to_plot if col in df.columns]
        if existing_columns:
            st.subheader("📈 بيانات الأسعار والمتوسطات")
            st.line_chart(df[existing_columns].tail(150))
        else:
            st.warning("لا توجد أعمدة صالحة للرسم البياني.")

        # -------------------------------
        # جدول إشارات التداول
        # -------------------------------
        if "Signal" in df.columns:
            st.subheader("🟢 إشارات التداول")
            st.dataframe(df[["Close","MACD","MACD_signal","RSI","Signal"]].tail(50))