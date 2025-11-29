import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from authlib.integrations.requests_client import OAuth2Session

# -------------------------
#       GitHub OAuth
# -------------------------
CLIENT_ID = st.secrets["GITHUB_CLIENT_ID"]
CLIENT_SECRET = st.secrets["GITHUB_CLIENT_SECRET"]
REDIRECT_URI = st.secrets["REDIRECT_URI"]

oauth = OAuth2Session(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    scope="read:user",
    redirect_uri=REDIRECT_URI
)

authorization_url, state = oauth.create_authorization_url(
    "https://github.com/login/oauth/authorize"
)

# -------------------------
#   Session state للتحقق من تسجيل الدخول
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# -------------------------
#       صفحة تسجيل الدخول
# -------------------------
if not st.session_state["logged_in"]:
    st.title("AI Smart Trader")
    st.write("### Login with GitHub")
    st.write(f"[Login with GitHub]({authorization_url})")
    
    # زر لتأكيد تسجيل الدخول بعد عملية OAuth
    if st.button("I have logged in"):
        # هنا يمكنك إضافة تحقق من الرمز إذا أردت
        st.session_state["logged_in"] = True
        st.experimental_rerun()  # إعادة تشغيل لتحديث الواجهة

# -------------------------
#       Dashboard بعد تسجيل الدخول
# -------------------------
if st.session_state["logged_in"]:
    st.title("AI Smart Trader Dashboard")

    # Sidebar
    st.sidebar.title("Settings")
    market_type = st.sidebar.selectbox("Select Market", ["Stocks", "Forex"])

    stocks_list = ["AAPL","TSLA","GOOGL","AMZN","MSFT","META","NVDA","NFLX"]
    forex_list = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD"]

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
        #       المؤشرات
        # -------------------------
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        RS = gain/loss
        df["RSI"] = 100 - (100/(1+RS))
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