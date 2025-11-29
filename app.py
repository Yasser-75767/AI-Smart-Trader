import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# -------------------------
#       إعداد الصفحة
# -------------------------
st.set_page_config(page_title="AI Smart Trader", layout="wide")
st.title("🎯 AI Smart Trader Pro — النسخة المستقرة مع إشارات التداول")

# -------------------------
#       اختيار السوق والرمز
# -------------------------
market_type = st.selectbox("اختر نوع السوق", ["أسهم", "فوركس"])

# خيارات الأسهم المشهورة
stocks_list = ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "META", "NFLX"]
# خيارات الفوركس المشهورة
forex_list = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

if market_type == "أسهم":
    symbol = st.selectbox("اختر السهم", stocks_list)
else:
    symbol = st.selectbox("اختر زوج الفوركس", forex_list) + "=X"  # صيغة ياهو فوركس

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("تاريخ البداية")
with col2:
    end_date = st.date_input("تاريخ النهاية")

chart_type = st.selectbox("اختر نوع الرسم", ["📉 الشموع اليابانية", "📈 الرسم الخطي"])

run = st.button("🔍 جلب البيانات وتحليلها")

# -------------------------
#       تحميل البيانات
# -------------------------
if run:
    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        st.error("❌ لم يتم العثور على بيانات!")
        st.stop()

    # -------------------------
    #       حساب المؤشرات
    # -------------------------
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    RS = gain / loss
    df["RSI"] = 100 - (100 / (1 + RS))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # إشارات BUY / SELL
    df["Buy"] = (df["SMA20"] > df["SMA50"]) & (df["MACD"] > df["Signal"])
    df["Sell"] = (df["SMA20"] < df["SMA50"]) & (df["MACD"] < df["Signal"])

    # -------------------------
    #       الرسم البياني
    # -------------------------
    st.subheader("📊 الرسم البياني مع الإشارات")

    fig = go.Figure()

    # الرسم حسب اختيار المستخدم
    if chart_type == "📉 الشموع اليابانية":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close"
        ))

    # SMA
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="orange")))

    # إشارات BUY
    buys = df[df["Buy"]]
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Close"],
        mode="markers+text",
        name="BUY",
        text=["BUY"]*len(buys),
        textposition="top center",
        marker=dict(color="green", size=12, symbol="triangle-up")
    ))

    # إشارات SELL
    sells = df[df["Sell"]]
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["Close"],
        mode="markers+text",
        name="SELL",
        text=["SELL"]*len(sells),
        textposition="bottom center",
        marker=dict(color="red", size=12, symbol="triangle-down")
    ))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    #       جدول البيانات
    # -------------------------
    st.subheader("📋 جدول البيانات والمؤشرات")
    st.dataframe(df.tail(200), height=600)