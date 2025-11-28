# نسخة أبسط تمامًا وبدون ta.trend
import yfinance as yf
import pandas as pd
import streamlit as st

st.title("🎯 AI Smart Trader Pro — نسخة بسيطة")

# --- إعدادات المستخدم ---
symbol = st.selectbox("اختر الأصل (رمز السهم)", ["AAPL","MSFT","GOOGL"])
start_date = st.date_input("تاريخ البداية", pd.to_datetime("2020-11-28"))
end_date = st.date_input("تاريخ النهاية", pd.to_datetime("2025-11-28"))

# --- جلب البيانات ---
df = yf.download(symbol, start=start_date, end=end_date)

if df.empty:
    st.error("لا توجد بيانات لهذا الرمز.")
else:
    # التأكد من أن Close 1D
    df['Close'] = df['Close'].values.flatten()

    # حساب المتوسطات البسيطة
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # إشارات شراء/بيع بسيطة
    df['Signal'] = 0
    df.loc[df['SMA_5'] > df['SMA_20'], 'Signal'] = 1  # شراء
    df.loc[df['SMA_5'] < df['SMA_20'], 'Signal'] = -1 # بيع

    # الأعمدة للرسم
    columns_to_plot = ['Close','SMA_5','SMA_20','EMA_10']

    st.subheader("📈 بيانات الأسعار والمتوسطات")
    st.line_chart(df[columns_to_plot].tail(150))

    st.subheader("💹 إشارات التداول")
    st.write(df[['Close','SMA_5','SMA_20','EMA_10','Signal']].tail(20))