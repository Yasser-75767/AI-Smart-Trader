import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# =========================
# واجهة التطبيق
# =========================
st.title("🎯 AI Smart Trader — نسخة الشموع اليابانية")

symbol = st.text_input("اختر الأصل (رمز السهم)", "AAPL")

start_date = st.date_input("تاريخ البداية")
end_date = st.date_input("تاريخ النهاية")

chart_type = st.selectbox(
    "اختر نوع الرسم:",
    ["الشموع اليابانية", "الرسم الخطي"]
)

# =========================
# تحميل البيانات
# =========================
if st.button("📥 جلب البيانات"):
    with st.spinner("جارِ تحميل البيانات..."):
        df = yf.download(symbol, start=start_date, end=end_date)

    if df is None or df.empty:
        st.error("❌ لم يتم العثور على بيانات!")
        st.stop()

    st.success("✅ تم تحميل البيانات بنجاح!")

    # =========================
    # عرض الجدول
    # =========================
    st.subheader("📊 جدول الأسعار")
    st.dataframe(df.tail(100))

    # =========================
    # رسم الشموع اليابانية
    # =========================
    if chart_type == "الشموع اليابانية":
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"]
                )
            ]
        )

        fig.update_layout(
            title="📉 الشموع اليابانية",
            xaxis_title="التاريخ",
            yaxis_title="السعر",
            template="plotly_dark",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # الرسم الخطي
    # =========================
    else:
        st.subheader("📈 الرسم الخطي للسعر")
        st.line_chart(df["Close"])