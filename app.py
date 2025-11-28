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
import time

st.set_page_config(page_title="AI Smart Trader Live 💜", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols
FEATURE_COLS = [
    "Open","High","Low","Close","Volume",
    "Price_Range","Price_Change","MA_5","Volume_MA"
]

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر سهم أو زوج الفوركس:", all_symbols)
uploaded_file = st.sidebar.file_uploader("ارفع صورة الشموع/المنحنيات للتحليل", type=["png","jpg","jpeg"])
update_seconds = st.sidebar.number_input("تحديث تلقائي بالثواني:", min_value=1, max_value=60, value=10)

# ===== وظائف أساسية =====
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        base_cols = ["Open","High","Low","Close","Volume"]
        if df.empty or not all(c in df.columns for c in base_cols):
            return pd.DataFrame()
        df = df[base_cols].dropna()
        return df
    except:
        return pd.DataFrame()

def prepare_features(df, with_target=True):
    df = df.copy()
    if len(df) < 2:
        return None, None, None

    df["Price_Range"] = df["High"] - df["Low"]
    df["Price_Change"] = df["Close"] - df["Open"]
    df["MA_5"] = df["Close"].rolling(5).mean().fillna(0)
    df["Volume_MA"] = df["Volume"].rolling(5).mean().fillna(0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    if with_target:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna(subset=["Target"])
        if df.empty:
            return None, None, None
        X = df[FEATURE_COLS]
        y = df["Target"].astype(int)
        return X, y, df
    else:
        X = df[FEATURE_COLS]
        return X, df, None

def train_model(df):
    X, y, _ = prepare_features(df)
    if X is None or len(X)<10:
        return None
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        tree_method="hist",
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X, y)
    return model

def predict_last(model, df):
    X_pred, _, _ = prepare_features(df, with_target=False)
    if X_pred is None or X_pred.empty:
        return None
    try:
        return model.predict(X_pred.iloc[[-1]].values)[0]
    except:
        return None

def analyze_image(file):
    try:
        image = Image.open(file).convert("RGB").resize((256,256))
        st.image(image, caption="📊 الصورة المحملة", use_column_width=True)
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        mean_val = float(np.mean(img_gray))
        st.write(f"📊 متوسط الإضاءة: {mean_val:.1f}")
        return 1 if mean_val>120 else 0
    except:
        return None

# ===== واجهة التطبيق =====
st.title("📈 AI Smart Trader Live 💜")
st.warning("⚠ التوصيات تعليمية فقط، التداول الحقيقي يحمل مخاطر مالية")

while True:
    start_date = datetime.date.today() - datetime.timedelta(days=60)
    end_date = datetime.date.today()

    df = load_data(symbol, start_date, end_date)
    if df.empty:
        st.error(f"⚠ لا توجد بيانات كافية للرمز {symbol}")
    else:
        model = train_model(df)
        if model is not None:
            pred = predict_last(model, df)
            if pred == 1:
                st.success(f"🔥 التنبؤ: {symbol} صاعد (تعليمي)")
            elif pred == 0:
                st.warning(f"📉 التنبؤ: {symbol} هابط (تعليمي)")
            else:
                st.info("⚠ لم يتمكن النموذج من التنبؤ")

        st.markdown("### آخر البيانات التاريخية:")
        st.dataframe(df.tail(5))
        st.markdown("### 📈 إحصائيات أساسية")
        col1,col2,col3=st.columns(3)
        with col1: st.metric("متوسط الإغلاق", f"{df['Close'].mean():.2f}")
        with col2: st.metric("أعلى سعر", f"{df['High'].max():.2f}")
        with col3: st.metric("أقل سعر", f"{df['Low'].min():.2f}")

        if uploaded_file is not None:
            st.markdown("### 📷 تحليل الصورة")
            img_pred = analyze_image(uploaded_file)
            if img_pred==1:
                st.success("🔥 تحليل الصورة: السوق يبدو صاعدًا")
            elif img_pred==0:
                st.warning("📉 تحليل الصورة: السوق يبدو هابطًا")

    st.markdown("---")
    st.subheader("⭐ رموز مقترحة (تعليميًا)")
    st.write(random.sample(all_symbols,5))

    st.info(f"⏱ التحديث التالي بعد {update_seconds} ثانية")
    time.sleep(update_seconds)
    st.experimental_rerun()