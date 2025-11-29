import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from authlib.integrations.requests_client import OAuth2Session

# -------------------------
#      إعداد OAuth GitHub
# -------------------------
CLIENT_ID = st.secrets["GITHUB_CLIENT_ID"]
CLIENT_SECRET = st.secrets["GITHUB_CLIENT_SECRET"]
REDIRECT_URI = st.secrets["REDIRECT_URI"]

AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
USER_API_URL = "https://api.github.com/user"

# -------------------------
#   Session state
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "user" not in st.session_state:
    st.session_state["user"] = None

# حسابات GitHub المسموح لها بالدخول
ALLOWED_USERS = ["yasser-75767"]  # عدّلها لو تحب تضيف مستخدمين آخرين


def show_login_page():
    """صفحة تسجيل الدخول"""
    st.title("AI Smart Trader")
    st.write("### Login with GitHub")

    # إنشاء رابط تسجيل الدخول
    oauth = OAuth2Session(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scope="read:user",
        redirect_uri=REDIRECT_URI,
    )

    authorization_url, state = oauth.create_authorization_url(AUTHORIZE_URL)

    # نعرض رابط تسجيل الدخول
    st.write(f"[🔑 Login with GitHub]({authorization_url})")


def handle_github_callback():
    """معالجة العودة من GitHub بعد تسجيل الدخول"""
    params = st.experimental_get_query_params()
    if "code" not in params:
        return  # مازال ما رجعش من GitHub

    code = params["code"][0]

    # نعمل OAuth session جديد علشان نجيب التوكن والمستخدم
    oauth = OAuth2Session(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
    )

    try:
        # الحصول على access token
        token = oauth.fetch_token(
            TOKEN_URL,
            code=code,
            client_secret=CLIENT_SECRET,
        )

        # جلب معلومات المستخدم
        resp = oauth.get(USER_API_URL)
        user_data = resp.json()
        username = user_data.get("login", None)

    except Exception:
        st.error("❌ Login failed. Please try again.")
        return

    if not username:
        st.error("❌ Could not get GitHub username.")
        return

    # التحقق هل المستخدم مسموح له أم لا
    if username not in ALLOWED_USERS:
        st.error("⚠️ Access denied. This GitHub account is not allowed.")
        return

    # تسجيل الدخول بنجاح
    st.session_state["logged_in"] = True
    st.session_state["user"] = username

    # تنظيف الرابط من ?code=...
    st.experimental_set_query_params()


def show_dashboard():
    """الواجهة الرئيسية بعد تسجيل الدخول"""
    st.title(f"AI Smart Trader Dashboard — Welcome {st.session_state['user']}")

    # -------------------------
    #       Sidebar
    # -------------------------
    st.sidebar.title("Settings")
    market_type = st.sidebar.selectbox("Select Market", ["Stocks", "Forex"])

    stocks_list = ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "META", "NVDA", "NFLX"]
    forex_list = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]

    if market_type == "Stocks":
        symbol_base = st.sidebar.selectbox("Select Stock", stocks_list)
        symbol = symbol_base
    else:
        symbol_base = st.sidebar.selectbox("Select Forex Pair", forex_list)
        symbol = symbol_base + "=X"

    # حماية اختيار الرمز
    if market_type == "Stocks" and symbol_base not in stocks_list:
        st.error("Invalid stock symbol.")
        st.stop()
    if market_type == "Forex" and symbol_base not in forex_list:
        st.error("Invalid forex pair.")
        st.stop()

    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])
    run = st.sidebar.button("Fetch Data & Analyze")

    if run:
        df = yf.download(symbol, start=start_date, end=end_date)

        if df.empty:
            st.error("No data found for this symbol and dates.")
            st.stop()

        # -------------------------
        #       المؤشرات
        # -------------------------
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()

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
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Candles",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    mode="lines",
                    name="Close",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA20"],
                name="SMA20",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA50"],
                name="SMA50",
            )
        )

        # إشارات BUY
        buys = df[df["Buy"]]
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Close"],
                mode="markers+text",
                name="BUY",
                text=["BUY"] * len(buys),
                textposition="top center",
                marker=dict(symbol="triangle-up", size=12),
            )
        )

        # إشارات SELL
        sells = df[df["Sell"]]
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["Close"],
                mode="markers+text",
                name="SELL",
                text=["SELL"] * len(sells),
                textposition="bottom center",
                marker=dict(symbol="triangle-down", size=12),
            )
        )

        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        #       جدول البيانات
        # -------------------------
        st.subheader("Data & Indicators (Last 200 rows)")
        st.dataframe(df.tail(200))


# =========================
#       تشغيل التطبيق
# =========================

# أولاً: لو مو مسجّل دخول، نحاول نشوف هل رجع من GitHub بـ ?code= أو لا
if not st.session_state["logged_in"]:
    handle_github_callback()

# لو بعد المعالجة مازال مو مسجّل → نعرض صفحة تسجيل الدخول
if not st.session_state["logged_in"]:
    show_login_page()
else:
    # مسجّل دخول ✅
    show_dashboard()