import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import random

st.set_page_config(page_title="AI Smart Trader", layout="wide")

# الأسهم البديلة الآمنة
fallback_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]


def load_data(symbol):
    """تنزيل البيانات والتحقق منها"""
    try:
        df = yf.download(symbol, period="3mo", interval="1d")
        required = ["Open", "High", "Low", "Close", "Volume"]

        if df.empty or not all(col in df.columns for col in required):
            st.warning(f"⚠ لا توجد بيانات كافية للسهم {symbol} — تم اختيار بديل.")
            alt = random.choice(fallback_symbols)
            df = yf.download(alt, period="3mo", interval="1d")

        return df

    except Exception:
        st.error("حدث خطأ أثناء تحميل البيانات.")
        alt = random.choice(fallback_symbols)
        return yf.download(alt, period="3mo", interval="1d")


def add_target(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()


def train_model(df):
    X = df[["Open", "High", "Low", "Close", "Volume"]]
    y = df["Target"]

    if len(df) < 5:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = xgb.XGBClassifier(n_estimators=80, max_depth=4)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


def predict(model, df):
    last = df.iloc[-1][["Open", "High", "Low", "Close", "Volume"]]
    last = np.array(last).reshape(1, -1)
    return model.predict(last)[0]


st.title("📈 AI Smart Trader — النسخة الثابتة بدون مشاكل")

symbol = st.text_input("أدخل رمز السهم", "AAPL")

if st.button("📊 الحصول على التوصيات"):

    st.info("⏳ يتم تحميل البيانات...")

    df = load_data(symbol)
    df = add_target(df)

    if df.empty:
        st.error("⚠ لا توجد بيانات كافية.")
        st.stop()

    model, acc = train_model(df)

    if model is None:
        st.error("⚠ البيانات غير كافية لتدريب النموذج.")
        st.stop()

    pred = predict(model, df)

    st.success(f"✔ دقة النموذج: {acc*100:.2f}%")

    if pred == 1:
        st.success("🔥 التنبؤ: السهم سيرتفع غدًا — شراء")
    else:
        st.warning("📉 التنبؤ: السهم سينخفض غدًا — بيع / تجنب")

    st.dataframe(df.tail(10))