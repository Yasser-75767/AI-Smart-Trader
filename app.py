import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.title("🎯 AI Smart Trader Pro — النسخة النهائية")
st.write("تحليل الأسهم باستخدام الذكاء الاصطناعي")

# --- الإعدادات ---
symbol = st.selectbox("اختر الأصل:", ["AAPL","MSFT","GOOGL","AMZN","TSLA"])
start_date = st.date_input("تاريخ البداية")
end_date = st.date_input("تاريخ النهاية")
lookback_min = st.number_input("أيام النظر للخلف (Min)", min_value=1, value=5)
lookback_max = st.number_input("أيام النظر للخلف (Max)", min_value=1, value=40)
confidence_threshold = st.slider("حد الثقة لإشارة قوية (%)", min_value=50, max_value=95, value=70)

# --- زر للحصول على النتائج ---
if st.button("الحصول على النتائج"):
    
    # تحميل البيانات
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        st.warning("لا توجد بيانات للأصل المختار في الفترة المحددة.")
    else:
        df.reset_index(inplace=True)
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # --- المؤشرات ---
        try:
            df["SMA_5"] = SMAIndicator(close, window=5).sma_indicator()
        except:
            df["SMA_5"] = np.nan
            st.warning("تعذر حساب SMA_5")

        try:
            df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
        except:
            df["SMA_20"] = np.nan
            st.warning("تعذر حساب SMA_20")

        try:
            df["EMA_10"] = EMAIndicator(close, window=10).ema_indicator()
        except:
            df["EMA_10"] = np.nan
            st.warning("تعذر حساب EMA_10")

        try:
            macd = MACD(close)
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
        except:
            df["MACD"] = np.nan
            df["MACD_signal"] = np.nan
            st.warning("تعذر حساب MACD")

        try:
            df["RSI"] = RSIIndicator(close).rsi()
        except:
            df["RSI"] = np.nan
            st.warning("تعذر حساب RSI")

        try:
            df["Volume_SMA"] = SMAIndicator(volume, window=20).sma_indicator()
            df["Volume_Ratio"] = volume / df["Volume_SMA"].replace(0,np.nan)
        except:
            df["Volume_Ratio"] = np.nan
            st.warning("تعذر حساب Volume Ratio")

        # --- عرض الرسم البياني للأعمدة الموجودة فقط ---
        columns_to_plot = [col for col in ["Close","SMA_5","SMA_20","EMA_10"] if col in df.columns]
        if columns_to_plot:
            st.line_chart(df[columns_to_plot].tail(150))
        
        # --- توليد إشارات تداول بسيطة ---
        signals = []
        for i in range(len(df)):
            if "MACD" in df.columns and "MACD_signal" in df.columns:
                if not pd.isna(df["MACD"].iloc[i]) and not pd.isna(df["MACD_signal"].iloc[i]):
                    if df["MACD"].iloc[i] > df["MACD_signal"].iloc[i]:
                        signals.append("شراء")
                    elif df["MACD"].iloc[i] < df["MACD_signal"].iloc[i]:
                        signals.append("بيع")
                    else:
                        signals.append("حيادي")
                else:
                    signals.append("غير متوفر")
            else:
                signals.append("غير متوفر")

        df["Signal"] = signals
        st.subheader("📊 إشارات التداول")
        st.dataframe(df[["Date","Close","Signal"]].tail(20))
        
        st.success("✅ تم التحليل بنجاح!")