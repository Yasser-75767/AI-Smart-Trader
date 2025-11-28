# AI Smart Trader Pro — نسخة مصححة
import yfinance as yf
import pandas as pd
import streamlit as st
from ta.trend import SMAIndicator, EMAIndicator

st.title("🎯 AI Smart Trader Pro — النسخة المصححة")

# --- إعدادات المستخدم ---
symbol = st.selectbox("اختر الأصل (رمز السهم)", ["AAPL","MSFT","GOOGL"])
start_date = st.date_input("تاريخ البداية", pd.to_datetime("2020-11-28"))
end_date = st.date_input("تاريخ النهاية", pd.to_datetime("2025-11-28"))
min_back = st.number_input("أيام النظر للخلف (Min)", min_value=1, value=5)
max_back = st.number_input("أيام النظر للخلف (Max)", min_value=1, value=40)
conf_min = st.slider("حد الثقة لإشارة قوية (Min %)", 0, 100, 0)
conf_max = st.slider("حد الثقة لإشارة قوية (Max %)", 0, 100, 100)

# --- جلب البيانات ---
df = yf.download(symbol, start=start_date, end=end_date)

if df.empty:
    st.error("لا توجد بيانات لهذا الرمز.")
else:
    # التأكد من أن Close هو Series 1D
    df['Close'] = df['Close'].squeeze()

    # حساب المؤشرات
    df['SMA_5'] = SMAIndicator(df['Close'], window=5).sma_indicator()
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA_10'] = EMAIndicator(df['Close'], window=10).ema_indicator()

    # إنشاء إشارات شراء وبيع بسيطة
    df['Signal'] = 0
    df.loc[df['SMA_5'] > df['SMA_20'], 'Signal'] = 1  # شراء
    df.loc[df['SMA_5'] < df['SMA_20'], 'Signal'] = -1 # بيع

    # الأعمدة للرسم
    columns_to_plot = [col for col in ['Close','SMA_5','SMA_20','EMA_10'] if col in df.columns]

    st.subheader("📈 بيانات الأسعار والمؤشرات")
    st.line_chart(df[columns_to_plot].tail(150))

    st.subheader("💹 إشارات التداول")
    st.write(df[['Close','SMA_5','SMA_20','EMA_10','Signal']].tail(20))