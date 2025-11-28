# app_live.py
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
from email.mime.multipart import MIMEMultipart
import time

st.set_page_config(page_title="AI Smart Trader Live 💜", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols
FEATURE_COLS = ["Open","High","Low","Close","Volume","Price_Range","Price_Change","MA_5","Volume_MA"]

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر سهم أو زوج الفوركس:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2023,1,1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
uploaded_file = st.sidebar.file_uploader("ارفع صورة الشموع/المنحنيات", type=["png","jpg","jpeg"])
refresh_sec = st.sidebar.number_input("تحديث تلقائي بالثواني:", min_value=5, max_value=60, value=10, step=1)

# إعداد البريد (اختياري)
st.sidebar.markdown("### تنبيهات البريد الإلكتروني (اختياري)")
smtp_server = st.sidebar.text_input("SMTP Server", value="smtp.gmail.com")
smtp_port = st.sidebar.number_input("SMTP Port", value=587)
sender_email = st.sidebar.text_input("البريد المرسل")
app_password = st.sidebar.text_input("كلمة مرور التطبيق", type="password")
receiver_email = st.sidebar.text_input("البريد المستلم")

# ===== وظائف التطبيق =====
def send_email(subject, body):
    if not all([smtp_server, smtp_port, sender_email, app_password, receiver_email]):
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        st.warning(f"⚠ خطأ عند إرسال البريد: {e}")

def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        base_cols = ["Open","High","Low","Close","Volume"]
        if df.empty or not all(c in df.columns for c in base_cols):
            return pd.DataFrame()
        df = df[base_cols].dropna()
        if len(df)<10: return pd.DataFrame()
        return df
    except:
        return pd.DataFrame()

def prepare_features(df, with_target=True):
    df = df.copy()
    base_cols = ["Open","High","Low","Close","Volume"]
    if not all(c in df.columns for c in base_cols): return None,None, None
    if with_target:
        df["Target"] = (df["Close"].shift(-1)>df["Close"]).astype(int)
        if df["Target"].dropna().empty: return None,None,None
    df["Price_Range"]=df["High"]-df["Low"]
    df["Price_Change"]=df["Close"]-df["Open"]
    df["MA_5"]=df["Close"].rolling(5).mean()
    df["Volume_MA"]=df["Volume"].rolling(5).mean()
    for col in FEATURE_COLS:
        if col not in df.columns: df[col]=0.0
    df[FEATURE_COLS]=df[FEATURE_COLS].fillna(0)
    if with_target:
        df=df.dropna(subset=["Target"])
        if df.empty: return None,None,None
        return df[FEATURE_COLS], df["Target"].astype(int), df
    else:
        return df[FEATURE_COLS], df, None

def train_model(df):
    X,y,_=prepare_features(df)
    if X is None or y is None or len(X)<30: return None,None
    split=int(len(X)*0.8)
    X_train,X_test=X[:split],X[split:]
    y_train,y_test=y[:split],y[split:]
    try:
        model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,
                                tree_method="hist",use_label_encoder=False,eval_metric="logloss",
                                random_state=42)
        model.fit(X_train,y_train)
        acc=accuracy_score(y_test,model.predict(X_test))
        return model, acc
    except:
        return None,None

def predict_last(model, df):
    X_pred, _, _ = prepare_features(df, with_target=False)
    if X_pred is None or X_pred.empty: return None
    try:
        return model.predict(X_pred.iloc[[-1]].values)[0]
    except:
        return None

def analyze_image(file):
    try:
        image=Image.open(file).convert("RGB").resize((256,256))
        st.image(image, caption="📊 الصورة المحملة", use_column_width=True)
        img_cv=np.array(image)
        img_gray=cv2.cvtColor(img_cv,cv2.COLOR_RGB2GRAY)
        mean_val=float(np.mean(img_gray))
        st.write(f"📊 متوسط الإضاءة في الصورة: {mean_val:.1f}")
        return 1 if mean_val>120 else 0
    except:
        return None

# ===== حلقة التحديث =====
def update():
    df = load_data(symbol, start_date, end_date)
    if df.empty:
        st.warning("⚠ البيانات غير كافية لهذا الرمز")
        return
    model, acc = train_model(df)
    if model is None:
        st.warning("⚠ النموذج لم يتم تدريبه بسبب قلة البيانات")
        return
    pred = predict_last(model, df)
    if pred is None:
        st.warning("⚠ لم يتمكن النموذج من التنبؤ")
        return

    # نتائج النص
    st.success(f"✔ دقة النموذج: {acc*100:.2f}%")
    if pred==1:
        msg=f"🔥 اتجاه {symbol} صاعد (إشارة شراء تعليمية)"
        st.success(msg)
        send_email(f"AI Smart Trader: {symbol} صعود", msg)
    else:
        msg=f"📉 اتجاه {symbol} هابط أو ضعيف (تجنب الشراء)"
        st.warning(msg)
        send_email(f"AI Smart Trader: {symbol} هبوط", msg)

    st.markdown("### آخر البيانات التاريخية:")
    st.dataframe(df.tail(10))

    # تحليل الصورة
    if uploaded_file:
        st.markdown("### 📷 تحليل الصورة")
        img_pred=analyze_image(uploaded_file)
        if img_pred==1: st.success("🔥 تحليل الصورة: السوق صاعد")
        elif img_pred==0: st.warning("📉 تحليل الصورة: السوق هابط أو ضعيف")
        else: st.info("⚠ لم يتمكن التطبيق من تحليل الصورة")

# ===== واجهة التطبيق =====
st.title("📈 AI Smart Trader Live 💜")
st.warning("⚠ التوصيات تعليمية فقط، التداول الحقيقي يحمل مخاطر مالية")

placeholder = st.empty()
with placeholder.container():
    update()

# التحديث التلقائي
while True:
    time.sleep(refresh_sec)
    placeholder.empty()
    with placeholder.container():
        update()