import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# -------------------------
#       إعداد الصفحة
# -------------------------
st.set_page_config(page_title="AI Smart Trader", layout="wide")
st.title("AI Smart Trader")  # فقط الاسم بدون أي كلمات إضافية

# -------------------------
#       اختيار السوق والرمز
# -------------------------
market_type = st.selectbox("Select Market", ["Stocks", "Forex"])

# قائمة كبيرة من الأسهم المشهورة (أكثر من 50)
stocks_list = [
    "AAPL","TSLA","GOOGL","AMZN","MSFT","META","NFLX","NVDA","JPM","BAC",
    "V","MA","DIS","ADBE","PYPL","INTC","CSCO","KO","PEP","NKE",
    "ORCL","CRM","WMT","T","VZ","BA","IBM","QCOM","MCD","SBUX",
    "GE","GM","F","AMD","SHOP","UBER","LYFT","SQ","TWTR","SNAP",
    "BIDU","JD","PDD","BABA","TCEHY","NIO","LI","XPEV","BYND","PLUG",
    "SPCE","RBLX","ZM","DOCU","ETSY","ROKU","ABNB","NET","OKTA","TEAM"
]

# قائمة كبيرة من أزواج الفوركس
forex_list = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
    "EURGBP","EURJPY","EURCHF","GBPJPY","AUDJPY","AUDNZD","CADJPY",
    "CHFJPY","GBPCHF","EURAUD","EURCAD","EURSEK","EURTRY",
    "GBPTRY","USDSGD","USDHKD","USDNOK","USDSEK","USDDKK","USDZAR",
    "USDTHB","USDINR"
]

# اختيار الرمز مع البحث
if market_type == "Stocks":
    symbol = st.selectbox("Select Stock", stocks_list, index=0)
else:
    symbol = st.selectbox("Select Forex Pair", forex_list, index=0) + "=X"

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date")
with col2:
    end_date = st.date_input("End Date")

chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"])

run = st.button("Fetch Data & Analyze")

# -------------------------
#       تحميل البيانات
# -------------------------
if run:
    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found for this symbol!")
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
    st.subheader("Chart with Buy/Sell Signals")

    fig = go.Figure()

    if chart_type == "Candlestick":
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
    st.subheader("Data & Indicators")
    st.dataframe(df.tail(200), height=600)