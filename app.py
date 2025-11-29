import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from authlib.integrations.requests_client import OAuth2Session
import os
from dotenv import load_dotenv

# -------------------------
#       تحميل إعدادات OAuth
# -------------------------
load_dotenv()
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")  # مثال: https://yourapp.streamlit.app

# -------------------------
#       إعداد OAuth
# -------------------------
oauth = OAuth2Session(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    scope="openid email profile",
    redirect_uri=REDIRECT_URI,
)

authorization_url, state = oauth.create_authorization_url(
    "https://accounts.google.com/o/oauth2/auth"
)

# -------------------------
#       تسجيل الدخول
# -------------------------
st.title("AI Smart Trader")

st.write("### Login with Google")
st.write(f"[Click here to login]({authorization_url})")

# بعد تسجيل الدخول يمكن إضافة استلام الـ token وبيانات المستخدم
# هنا ستضع باقي تطبيقك بعد تسجيل الدخول

if st.button("Continue to Dashboard (mock)"):
    # -------------------------
    #       واجهة Sidebar
    # -------------------------
    st.sidebar.title("Settings")

    market_type = st.sidebar.selectbox("Select Market", ["Stocks", "Forex"])

    stocks_list = [
        "AAPL","TSLA","GOOGL","AMZN","MSFT","META","NFLX","NVDA","JPM","BAC",
        "V","MA","DIS","ADBE","PYPL","INTC","CSCO","KO","PEP","NKE",
        "ORCL","CRM","WMT","T","VZ","BA","IBM","QCOM","MCD","SBUX",
        "GE","GM","F","AMD","SHOP","UBER","LYFT","SQ","TWTR","SNAP",
        "BIDU","JD","PDD","BABA","TCEHY","NIO","LI","XPEV","BYND","PLUG",
        "SPCE","RBLX","ZM","DOCU","ETSY","ROKU","ABNB","NET","OKTA","TEAM"
    ]

    forex_list = [
        "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
        "EURGBP","EURJPY","EURCHF","GBPJPY","AUDJPY","AUDNZD","CADJPY",
        "CHFJPY","GBPCHF","EURAUD","EURCAD","EURSEK","EURTRY",
        "GBPTRY","USDSGD","USDHKD","USDNOK","USDSEK","USDDKK","USDZAR",
        "USDTHB","USDINR"
    ]

    if market_type == "Stocks":
        symbol = st.sidebar.selectbox("Select Stock", stocks_list)
    else:
        symbol = st.sidebar.selectbox("Select Forex Pair", forex_list) + "=X"

    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])
    run = st.sidebar.button("Fetch Data & Analyze")

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

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        RS = gain / loss
        df["RSI"] = 100 - (100 / (1 + RS))

        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

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

        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="orange")))

        buys = df[df["Buy"]]
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Close"],
            mode="markers+text",
            name="BUY",
            text=["BUY"]*len(buys),
            textposition="top center",
            marker=dict(color="green", size=12, symbol="triangle-up")
        ))

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

        st.subheader("Data & Indicators")
        st.dataframe(df.tail(200), height=600)