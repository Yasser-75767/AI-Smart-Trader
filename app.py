import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.title("🎯 AI Smart Trader Pro — النسخة النهائية مع إشارات التداول")

# ---- واجهة المستخدم ----
symbol = st.selectbox("اختر الأصل:", ["AAPL","GOOGL","MSFT","TSLA"])
start_date = st.date_input("تاريخ البداية")
end_date = st.date_input("تاريخ النهاية")
lookback_min = st.number_input("أيام النظر للخلف (Min)", min_value=1, max_value=100, value=5)
lookback_max = st.number_input("أيام النظر للخلف (Max)", min_value=lookback_min, max_value=100, value=40)
confidence_min = st.slider("حد الثقة لإشارة قوية (Min %)", 0, 100, 50)
confidence_max = st.slider("حد الثقة لإشارة قوية (Max %)", 0, 100, 95)

# ---- زر الحصول على النتائج ----
if st.button("📈 الحصول على النتائج"):

    # تحميل البيانات
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        st.error("لا توجد بيانات متاحة لهذا النطاق الزمني.")
    else:
        # ---- حساب المؤشرات ----
        try:
            df['SMA_5'] = SMAIndicator(df['Close'], window=5).sma_indicator().squeeze()
            df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator().squeeze()
            df['EMA_10'] = EMAIndicator(df['Close'], window=10).ema_indicator().squeeze()
            macd = MACD(df['Close'])
            df['MACD'] = macd.macd().squeeze()
            df['MACD_signal'] = macd.macd_signal().squeeze()
            df['RSI'] = RSIIndicator(df['Close']).rsi().squeeze()
            df['Volume_SMA'] = SMAIndicator(df['Volume'], window=20).sma_indicator().squeeze()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].replace(0,np.nan)
        except Exception as e:
            st.error(f"خطأ في حساب المؤشرات: {e}")

        # ---- تأكد من وجود الأعمدة قبل الرسم ----
        columns_to_plot = [c for c in ['Close','SMA_5','SMA_20','EMA_10'] if c in df.columns]
        if columns_to_plot:
            st.subheader("📊 بيانات الأسعار والمتوسطات")
            st.line_chart(df[columns_to_plot].tail(150))
        else:
            st.warning("لا توجد أعمدة للرسم.")

        # ---- عرض جدول بيانات مختصر ----
        st.subheader("📋 بيانات مختصرة")
        st.dataframe(df.tail(10))