import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.title("🎯 AI Smart Trader — الرسم و الشموع اليابانية")

# --- اختيار السهم ---
symbol = st.text_input("أدخل رمز السهم", "AAPL")

# --- اختيار التواريخ ---
start_date = st.date_input("تاريخ البداية", pd.to_datetime("2020-11-28"))
end_date = st.date_input("تاريخ النهاية", pd.to_datetime("2025-11-28"))

# --- تحميل البيانات ---
df = yf.download(symbol, start=start_date, end=end_date)

if df.empty:
    st.error("⚠ لا توجد بيانات لهذا السهم أو التاريخ غير صحيح")
    st.stop()

# تجهيز البيانات
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.reset_index(inplace=True)

# --- اختيار نوع الرسم ---
chart_type = st.radio("اختر نوع الرسم", ["الشموع اليابانية", "الرسم البياني العادي"])

st.write("### 📊 الرسم:")

# --- الشموع اليابانية ---
if chart_type == "الشموع اليابانية":
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=f"Candlestick — {symbol}", xaxis_title="التاريخ", yaxis_title="السعر")
    st.plotly_chart(fig, use_container_width=True)

# --- الرسم البياني العادي ---
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close'
    ))
    fig.update_layout(title=f"Line Chart — {symbol}", xaxis_title="التاريخ", yaxis_title="السعر")
    st.plotly_chart(fig, use_container_width=True)

# --- جدول البيانات ---
st.write("### 📁 جدول الأسعار (آخر 100 يوم)")
st.dataframe(df.tail(100))