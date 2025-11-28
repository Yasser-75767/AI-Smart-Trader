import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.title("🎯 AI Smart Trader Pro — نسخة محسنة (اختيار الرسم)")

# اختيار السهم
symbol = st.selectbox("اختر الأصل (رمز السهم)", ["AAPL","MSFT","GOOGL"])
start_date = st.date_input("تاريخ البداية", pd.to_datetime("2020-11-28"))
end_date = st.date_input("تاريخ النهاية", pd.to_datetime("2025-11-28"))

# اختيار نوع الرسم
chart_type = st.radio("اختر نوع الرسم:", ("الشموع اليابانية", "الرسم البياني العادي"))

# تحميل البيانات
df = yf.download(symbol, start=start_date, end=end_date)

if df.empty:
    st.error("لا توجد بيانات لهذا الرمز.")
else:
    # إعادة تشكيل العمود Close لتجنب أي خطأ
    df['Close'] = df['Close'].values.flatten()

    if chart_type == "الرسم البياني العادي":
        st.subheader("📈 أسعار الإغلاق")
        st.line_chart(df['Close'])
    else:
        st.subheader("🕯 الشموع اليابانية")
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        st.plotly_chart(fig)