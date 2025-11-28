# app_streamlit_final.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from PIL import Image
import cv2
import random
import datetime
import smtplib
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader Live 💜", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols
FEATURE_COLS = ["Open","High","Low","Close","Volume","Price_Range","Price_Change","MA_5","Volume_MA"]

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر سهم أو زوج الفوركس:", all_symbols)
uploaded_file = st.sidebar.file_uploader("رفع صورة الشموع/المنحنيات", type=["png","jpg","jpeg"])
refresh_rate = st.sidebar.slider("تحديث تلقائي بالثواني", 1, 10, 3)

# إعداد البريد (اختياري)
st.sidebar.markdown("---")
st.sidebar.subheader("تنبيهات البريد الإلكتروني (اختياري)")
smtp_server = st.sidebar.text_input("SMTP Server", "smtp.gmail.com")
smtp_port = st.sidebar.number_input("SMTP Port", 587)
email_sender = st.sidebar.text_input("البريد المرسل")
email_password = st.sidebar.text_input("كلمة مرور التطبيق", type="password")
email_receiver = st.sidebar.text_input("البريد المستلم")

# ===== دوال التطبيق =====
def send_email(subject, message):
    if not all([smtp_server, smtp_port, email_sender, email_password, email_receiver]):
        return
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = email_sender
        msg["To"] = email_receiver
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_sender, email_password)
            server.send_message(msg)
    except Exception as e:
        st.warning(f"⚠ خطأ في إرسال البريد: {e}")

def load_data(symbol, period="60d"):
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    df = df[["Open","High","Low","Close","Volume"]].fillna(0)
    return df

def prepare_features(df):
    df = df.copy()
    df["Price_Range"] = df["High"]-df["Low"]
    df["Price_Change"] = df["Close"]-df["Open"]
    df["MA_5"] = df["Close"].rolling(5).mean().fillna(0)
    df["Volume_MA"] = df["Volume"].rolling(5).mean().fillna(0)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=["Target"])
    X = df[FEATURE_COLS]; y = df["Target"].astype(int)
    return X, y, df

def train_predict(df):
    X, y, _ = prepare_features(df)
    if len(X)<10: return None, None
    split=int(len(X)*0.8)
    X_train,X_test=X[:split],X[split:]
    y_train,y_test=y[:split],y[split:]
    model = xgb.XGBClassifier(n_estimators=50,max_depth=3,learning_rate=0.1,
                              tree_method="hist",use_label_encoder=False,eval_metric="logloss")
    model.fit(X_train,y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    last_row = X.iloc[[-1]].values
    pred = model.predict(last_row)[0]
    return acc, pred

def analyze_image(file):
    try:
        img = Image.open(file).convert("RGB").resize((128,128))
        st.image(img, caption="📊 الصورة", use_column_width=True)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        return 1 if np.mean(gray)>120 else 0
    except: return None

# ===== واجهة التطبيق =====
st.title("📈 AI Smart Trader Live 💜")
st.warning("⚠ التوصيات تعليمية فقط، التداول الحقيقي يحمل مخاطر مالية")

placeholder = st.empty()
last_pred = None

def update():
    global last_pred
    with placeholder.container():
        df = load_data(symbol)
        with ThreadPoolExecutor() as executor:
            future = executor.submit(train_predict, df)
            future_img = executor.submit(analyze_image, uploaded_file) if uploaded_file else None
            acc, pred = future.result()
            img_pred = future_img.result() if future_img else None

        st.subheader(f"📊 {symbol} — تحديث مباشر")
        st.write(df.tail(5))
        st.success(f"✔ دقة النموذج: {acc*100:.2f}%")
        st.info("🔥 صاعد" if pred==1 else "📉 هابط/ضعيف")

        if last_pred is not None and pred != last_pred:
            send_email(f"تغير اتجاه {symbol}", f"اتجاه {symbol} تغير من {last_pred} إلى {pred}")
        last_pred = pred

        if img_pred is not None:
            if img_pred==1: st.success("🔥 الصورة تشير لصعود السوق")
            elif img_pred==0: st.warning("📉 الصورة تشير لهبوط السوق")

        st.markdown("---")
        st.write("⭐ رموز للمراقبة تعليمياً")
        st.write(random.sample(all_symbols,5))

# حلقات التحديث
import time
while True:
    update()
    time.sleep(refresh_rate)