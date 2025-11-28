# app_pro.py — AI Smart Trader Pro (نسخة قوية بدون مكتبات ثقيلة مثل xgboost و ta)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import datetime
import warnings
warnings.filterwarnings("ignore")
import random

# ================= إعداد الصفحة =================
st.set_page_config(page_title="AI Smart Trader Pro — النسخة الاحترافية", layout="wide")

# ================= القوائم =================
stocks = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN",
    "TSLA", "META", "JPM", "JNJ", "V",
    "WMT", "PG", "DIS", "NFLX", "ADBE"
]

forex_pairs = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"
]

crypto = [
    "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LTC-USD",
    "XRP-USD", "DOGE-USD", "SOL-USD", "AVAX-USD", "MATIC-USD"
]

all_symbols = stocks + forex_pairs + crypto

# ================= الشريط الجانبي =================
st.sidebar.header("⚙️ الإعدادات")

symbol = st.sidebar.selectbox("اختر الأصل:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())

model_type = st.sidebar.selectbox(
    "نموذج الذكاء الاصطناعي:",
    ["Random Forest", "Gradient Boosting", "Neural Network (MLP)", "Ensemble (RF + GB + MLP)"]
)

test_size = st.sidebar.slider("حجم بيانات الاختبار (%):", 10, 40, 20)
confidence_threshold = st.sidebar.slider("حد الثقة لإشارة قوية (%):", 50, 95, 75)

# ================= دوال المساعدة =================

def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty or len(df) < 200:
        return pd.DataFrame()
    return df

def add_indicators(df):
    df = df.copy().dropna()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # عوائد
    df["Return"] = close.pct_change()
    df["LogReturn"] = np.log(close / close.shift(1))

    # متوسطات متحركة
    df["SMA_5"] = close.rolling(5).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()

    # RSI يدوي (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd - macd_signal

    # Bollinger Bands (20)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Middle"] = ma20
    df["BB_Upper"] = ma20 + 2 * std20
    df["BB_Lower"] = ma20 - 2 * std20
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]

    # تقلب وحجم
    df["Volatility_20"] = df["Return"].rolling(20).std()
    df["Volume_MA20"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / df["Volume_MA20"].replace(0, np.nan)

    # نطاق السعر والفجوة
    df["Price_Range"] = (high - low) / close
    df["Gap"] = (df["Open"] - close.shift(1)) / close.shift(1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def create_target(df, lookahead=1):
    future_close = df["Close"].shift(-lookahead)
    future_return = future_close / df["Close"] - 1
    df["Target"] = (future_return > 0).astype(int)
    df = df.dropna()
    return df

def prepare_features(df):
    feature_cols = [
        "Close", "SMA_5", "SMA_20", "SMA_50",
        "EMA_10", "EMA_50",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Lower", "BB_Width",
        "Return", "LogReturn",
        "Volatility_20", "Volume_Ratio",
        "Price_Range", "Gap"
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available]
    y = df["Target"]
    return X, y, available

def train_model(X, y, model_type, test_ratio):
    if len(X) < 150:
        return None, None, "بيانات غير كافية بعد المعالجة"

    split_point = int(len(X) * (1 - test_ratio))
    if split_point <= 0 or split_point >= len(X) - 1:
        return None, None, "مشكلة في تقسيم البيانات"

    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )

    if model_type == "Random Forest":
        model = rf
    elif model_type == "Gradient Boosting":
        model = gb
    elif model_type == "Neural Network (MLP)":
        model = mlp
    else:
        model = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("mlp", mlp)],
            voting="soft"
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, "نجاح"

def generate_signal(model, X_last, last_row, conf_threshold):
    proba_up = model.predict_proba(X_last)[0][1]
    pred = int(proba_up > 0.5)

    rsi = float(last_row.get("RSI_14", 50))
    macd = float(last_row.get("MACD", 0))
    volume_ratio = float(last_row.get("Volume_Ratio", 1))

    if pred == 1 and proba_up * 100 >= conf_threshold:
        if rsi < 70 and macd > 0 and volume_ratio > 1:
            signal = "شراء قوي 🟢"
        else:
            signal = "شراء 🟢"
    elif pred == 0 and (1 - proba_up) * 100 >= conf_threshold:
        if rsi > 30 and macd < 0 and volume_ratio > 1:
            signal = "بيع قوي 🔴"
        else:
            signal = "بيع 🔴"
    else:
        signal = "محايد ⚪"

    confidence = max(proba_up, 1 - proba_up)
    return signal, confidence

def calculate_risk_metrics(df):
    returns = df["Close"].pct_change().dropna()
    if returns.empty:
        return {}
    metrics = {
        "العائد اليومي المتوسط": f"{returns.mean()*100:.2f}%",
        "التقلب اليومي": f"{returns.std()*100:.2f}%",
        "أقصى خسارة يومية": f"{returns.min()*100:.2f}%",
        "أقصى ربح يومي": f"{returns.max()*100:.2f}%",
        "الانحراف المعياري": f"{returns.std()*100:.2f}%",
    }
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        metrics["معدل شارب التقريبي"] = f"{sharpe:.2f}"
    else:
        metrics["معدل شارب التقريبي"] = "N/A"
    return metrics

def fetch_market_sentiment():
    return {
        "bullish": random.uniform(0.4, 0.7),
        "bearish": random.uniform(0.2, 0.6),
        "neutral": random.uniform(0.1, 0.3),
    }

# ================= واجهة التطبيق =================

st.title("🤖 AI Smart Trader Pro — النسخة الاحترافية B")
st.caption("تطبيق تحليلي متقدم يمكنك الاعتماد عليه كأداة مساعدة في قراراتك.")

if st.button("🚀 بدء التحليل المتقدم"):
    with st.spinner("جاري تحميل البيانات وتحليل السوق..."):
        df = load_data(symbol, start_date, end_date)

        if df.empty:
            st.error("❌ لا توجد بيانات كافية (يجب أن يكون هناك على الأقل ~200 شمعة).")
        else:
            df = add_indicators(df)
            df = create_target(df)
            X, y, feat_cols = prepare_features(df)

            model, acc, status = train_model(X, y, model_type, test_size/100)

            if model is None:
                st.error(f"❌ فشل في تدريب النموذج: {status}")
            else:
                st.success(f"✅ دقة النموذج على البيانات التاريخية: **{acc*100:.2f}%**")

                # التنبؤ الحالي
                last_X = X.iloc[[-1]]
                last_row = df.iloc[-1]
                signal, confidence = generate_signal(
                    model, last_X, last_row, confidence_threshold
                )

                col_sig, col_chart = st.columns([1, 2])

                with col_sig:
                    st.subheader("🎯 إشارة التداول الحالية")
                    conf_str = f"ثقة: {confidence*100:.1f}%"

                    if "شراء قوي" in signal:
                        st.success(f"{signal}\n\n{conf_str}")
                    elif "بيع قوي" in signal:
                        st.error(f"{signal}\n\n{conf_str}")
                    elif "شراء" in signal:
                        st.info(f"{signal}\n\n{conf_str}")
                    elif "بيع" in signal:
                        st.warning(f"{signal}\n\n{conf_str}")
                    else:
                        st.warning(f"{signal}\n\n{conf_str}")

                    st.caption("هذه الإشارة ناتجة عن نموذج ML + RSI + MACD + حجم التداول.")

                with col_chart:
                    st.subheader("📈 السعر مع المتوسطات المتحركة")
                    plot_df = df[["Close", "SMA_20", "SMA_50"]].tail(150)
                    st.line_chart(plot_df)

                # Tabs للتحليل المتقدم
                st.markdown("---")
                st.subheader("📊 تحليل متقدم للحالة الحالية")

                tab1, tab2, tab3, tab4 = st.tabs(
                    ["المؤشرات الفنية", "مقاييس المخاطرة", "تحليل مشاعر السوق", "البيانات التاريخية"]
                )

                rsi_val = float(last_row.get("RSI_14", 50))
                macd_val = float(last_row.get("MACD", 0))
                vol_ratio = float(last_row.get("Volume_Ratio", 1))

                with tab1:
                    st.write("**أهم المؤشرات الآن:**")
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        delta = (
                            "تشبع شراء" if rsi_val > 70 else
                            "تشبع بيع" if rsi_val < 30 else
                            "محايد"
                        )
                        st.metric("RSI (14)", f"{rsi_val:.1f}", delta=delta)

                    with c2:
                        delta = "زخم صاعد" if macd_val > 0 else "زخم هابط"
                        st.metric("MACD", f"{macd_val:.4f}", delta=delta)

                    with c3:
                        delta = "حجم مرتفع" if vol_ratio > 1.5 else "حجم منخفض"
                        st.metric("Volume Ratio", f"{vol_ratio:.2f}", delta=delta)

                with tab2:
                    st.write("**مقاييس المخاطرة:**")
                    risk = calculate_risk_metrics(df)
                    if not risk:
                        st.info("لا يمكن حساب المقاييس حالياً.")
                    else:
                        for k, v in risk.items():
                            st.metric(k, v)

                with tab3:
                    st.write("**تحليل مشاعر (محاكاة بسيطة):**")
                    sentiment = fetch_market_sentiment()
                    s1, s2, s3 = st.columns(3)
                    s1.metric("صاعد (Bullish)", f"{sentiment['bullish']*100:.1f}%")
                    s2.metric("هابط (Bearish)", f"{sentiment['bearish']*100:.1f}%")
                    s3.metric("محايد (Neutral)", f"{sentiment['neutral']*100:.1f}%")

                with tab4:
                    st.write("**آخر 50 شمعة:**")
                    st.dataframe(df.tail(50))

st.markdown("---")
st.caption("أنت المسؤولة عن قراراتك، والتطبيق مجرد أداة تحليل وتصفية ذكية للفرص 🔍.")