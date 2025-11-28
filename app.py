import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ta
import datetime

# ---------------------------------------------------------
#        تحميل البيانات + المؤشرات الفنية (نسخة آمنة)
# ---------------------------------------------------------
def load_enhanced_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)

    if df.empty:
        return df

    # تحويل الأعمدة لسلاسل 1D
    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # مؤشرات MA (بدون أخطاء)
    df["SMA_5"] = close.rolling(5).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()

    # مؤشر MACD
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # الانحراف المعياري للتذبذب
    df["Volatility"] = close.rolling(10).std()

    # الفوليوم نسبة
    df["Volume_SMA"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / df["Volume_SMA"].replace(0, np.nan)

    # الفجوة السعرية
    df["Gap"] = (open_ - close.shift(1)) / close.shift(1)

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------
#                تجهيز البيانات للنموذج
# ---------------------------------------------------------
def prepare_ml_data(df, lookback):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    features = [
        "SMA_5", "SMA_20", "SMA_50",
        "MACD", "MACD_Signal", "MACD_Hist",
        "RSI", "Volatility",
        "Volume_Ratio", "Gap"
    ]

    X = df[features]
    y = df["Target"]

    return train_test_split(X, y, test_size=0.2, shuffle=False)


# ---------------------------------------------------------
#                       واجهة التطبيق
# ---------------------------------------------------------
st.title("🎯 AI Smart Trader Pro — النسخة النهائية")

st.sidebar.header("⚙️ الإعدادات المتقدمة")
symbol = st.sidebar.text_input("اختر الأصل:", "AAPL")

start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())

lookback = st.sidebar.slider("أيام النظر للخلف:", 5, 40, 10)
confidence_limit = st.sidebar.slider("حد الثقة لإشارة قوية (%):", 50, 95, 70)

# ---------------------------------------------------------
#             تحميل البيانات + تدريب النموذج
# ---------------------------------------------------------
df = load_enhanced_data(symbol, start_date, end_date)

if df.empty:
    st.error("لم يتم العثور على بيانات. جرّب رمزًا آخر.")
    st.stop()

X_train, X_test, y_train, y_test = prepare_ml_data(df, lookback)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

st.write(f"✅ **دقة النموذج: {acc*100:.2f}%**")

# ---------------------------------------------------------
#                    إشارة التداول
# ---------------------------------------------------------
last_row = df.tail(1)
last_features = last_row[[
    "SMA_5", "SMA_20", "SMA_50",
    "MACD", "MACD_Signal", "MACD_Hist",
    "RSI", "Volatility",
    "Volume_Ratio", "Gap"
]]

proba = model.predict_proba(last_features)[0][1] * 100

st.subheader("🎯 إشارة التداول الحالية")

if proba > confidence_limit:
    st.success(f"📈 شراء — الثقة: {proba:.2f}%")
elif proba < (100 - confidence_limit):
    st.error(f"📉 بيع — الثقة: {proba:.2f}%")
else:
    st.warning(f"⚠️ محايد — الثقة: {proba:.2f}%")


# ---------------------------------------------------------
#                    عرض الرسم البياني
# ---------------------------------------------------------
st.subheader("📈 السعر + المتوسطات المتحركة")

df_plot = df[["Close", "SMA_5", "SMA_20", "SMA_50"]].tail(200)
st.line_chart(df_plot)