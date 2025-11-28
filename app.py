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
from streamlit_autorefresh import st_autorefresh

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader Live 💜", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Price_Range", "Price_Change", "MA_5", "Volume_MA"
]

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر سهم أو زوج الفوركس:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
uploaded_file = st.sidebar.file_uploader("ارفع صورة الشموع/المنحنيات", type=["png","jpg","jpeg"])
update_sec = st.sidebar.number_input("تحديث تلقائي بالثواني", min_value=1, max_value=60, value=10)

# ===== تحديث تلقائي =====
st_autorefresh(interval=update_sec*1000, limit=None, key="autorefresh")

# ===== تحميل البيانات =====
@st.cache_data(show_spinner=False)
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        base_cols = ["Open", "High", "Low", "Close", "Volume"]
        if df.empty or not all(c in df.columns for c in base_cols):
            return pd.DataFrame()
        return df[base_cols].dropna()
    except:
        return pd.DataFrame()

# ===== تجهيز الميزات =====
def prepare_features(df, with_target=True):
    df = df.copy()
    if df.empty or len(df)<10:
        return None, None, None

    # الهدف
    if with_target:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # الميزات
    df["Price_Range"] = df["High"] - df["Low"]
    df["Price_Change"] = df["Close"] - df["Open"]
    df["MA_5"] = df["Close"].rolling(5).mean().fillna(0)
    df["Volume_MA"] = df["Volume"].rolling(5).mean().fillna(0)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    if with_target:
        df = df.dropna(subset=["Target"])
        if df.empty:
            return None, None, None
        X = df[FEATURE_COLS]
        y = df["Target"].astype(int)
        return X, y, df
    else:
        X = df[FEATURE_COLS]
        return X, df, None

# ===== تدريب النموذج =====
def train_model(df):
    X, y, _ = prepare_features(df)
    if X is None or len(X)<30:
        return None, None
    split_point = int(len(X)*0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    try:
        model = xgb.XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            tree_method="hist",
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return model, acc
    except:
        return None, None

# ===== التنبؤ =====
def predict_last(model, df):
    X_pred, _, _ = prepare_features(df, with_target=False)
    if X_pred is None or X_pred.empty:
        return None
    last_row = X_pred.iloc[[-1]].values
    try:
        return model.predict(last_row)[0]
    except:
        return None

# ===== تحليل الصورة =====
def analyze_image(file):
    try:
        image = Image.open(file).convert("RGB").resize((256,256))
        st.image(image, caption="📷 الصورة", use_column_width=True)
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        return 1 if np.mean(img_gray)>120 else 0
    except:
        return None

# ===== واجهة التطبيق =====
st.title("📈 AI Smart Trader Live 💜")
st.warning("⚠ التوصيات تعليمية فقط، التداول يحمل مخاطر مالية")

df = load_data(symbol, start_date, end_date)
if df.empty:
    st.error("⚠ لا توجد بيانات كافية لهذا الرمز")
else:
    model, acc = train_model(df)
    if model is None:
        st.warning("⚠ النموذج لم يتم تدريبه بسبب قلة البيانات")
    else:
        pred = predict_last(model, df)
        st.success(f"✔ دقة النموذج: {acc*100:.2f}%")
        if pred==1:
            st.success(f"🔥 التنبؤ: {symbol} صاعد (شراء تعليمي)")
        else:
            st.warning(f"📉 التنبؤ: {symbol} هابط/ضعيف")

    # عرض البيانات الأخيرة
    st.markdown("### آخر البيانات التاريخية")
    st.dataframe(df.tail(10))

    # إحصائيات أساسية
    col1, col2, col3 = st.columns(3)
    col1.metric("متوسط الإغلاق", f"{df['Close'].mean():.2f}")
    col2.metric("أعلى سعر", f"{df['High'].max():.2f}")
    col3.metric("أقل سعر", f"{df['Low'].min():.2f}")

    # تحليل الصورة إن وُجدت
    if uploaded_file is not None:
        img_pred = analyze_image(uploaded_file)
        if img_pred==1:
            st.success("🔥 تحليل الصورة: السوق صاعد")
        elif img_pred==0:
            st.warning("📉 تحليل الصورة: السوق هابط/ضعيف")
        else:
            st.info("⚠ لم يتمكن التطبيق من تحليل الصورة")

# رموز عشوائية تعليمية
st.markdown("---")
st.subheader("⭐ رموز للمراقبة (تعليميًا)")
st.write(random.sample(all_symbols, 5))