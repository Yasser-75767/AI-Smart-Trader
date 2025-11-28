# app_fast.py
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
import threading

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader Live 💜", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols

FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume",
                "Price_Range", "Price_Change", "MA_5", "Volume_MA"]

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر سهم أو زوج الفوركس:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
uploaded_file = st.sidebar.file_uploader("ارفع صورة الشموع/المنحنيات للتحليل", type=["png","jpg","jpeg"])
update_sec = st.sidebar.slider("تحديث تلقائي بالثواني", 1, 30, 10)

# ===== تحميل البيانات مع بديل =====
def load_data_with_fallback(original_symbol, start, end):
    candidates = [original_symbol] + [s for s in all_symbols if s != original_symbol]
    for sym in candidates:
        try:
            df = yf.download(sym, start=start, end=end, progress=False)
        except Exception:
            continue
        base_cols = ["Open", "High", "Low", "Close", "Volume"]
        if df.empty or not all(c in df.columns for c in base_cols):
            continue
        df = df[base_cols].dropna()
        if len(df) < 10:
            continue
        if sym != original_symbol:
            st.info(f"ℹ تم استخدام الرمز البديل: {sym} بدل {original_symbol}")
        return df, sym
    return pd.DataFrame(), original_symbol

# ===== تجهيز الميزات =====
def prepare_features(df, with_target=True):
    df = df.copy()
    if df.empty or len(df)<2:
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

# ===== تدريب النموذج =====
def train_model(df, result):
    X, y, df_feat = prepare_features(df, with_target=True)
    if X is None or y is None or len(X)<30:
        result["model"], result["acc"] = None, None
        return
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    try:
        model = xgb.XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.1,
            tree_method="hist", use_label_encoder=False, eval_metric="logloss", random_state=42
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        result["model"], result["acc"] = model, acc
    except:
        result["model"], result["acc"] = None, None

# ===== التنبؤ =====
def predict_last(model, df):
    X_pred, df_feat, _ = prepare_features(df, with_target=False)
    if X_pred is None or X_pred.empty:
        return None
    last_row = X_pred.iloc[[-1]].values
    try:
        return model.predict(last_row)[0]
    except:
        return None

# ===== تحليل الصور =====
def analyze_image(file):
    try:
        image = Image.open(file).convert("RGB").resize((256,256))
        st.image(image, caption="📊 الصورة المحملة", use_column_width=True)
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        mean_val = float(np.mean(img_gray))
        st.write(f"📊 متوسط الإضاءة في الصورة: {mean_val:.1f}")
        return 1 if mean_val>120 else 0
    except:
        return None

# ===== واجهة التطبيق =====
st.title("📈 AI Smart Trader Live 💜")
st.warning("⚠ التوصيات تعليمية فقط، التداول الحقيقي يحمل مخاطر مالية")

def update():
    with st.spinner("⏳ جاري التحليل..."):
        df, used_symbol = load_data_with_fallback(symbol, start_date, end_date)
        if df.empty:
            st.error("⚠ لا توجد بيانات كافية لهذا الرمز أو البدائل.")
            return
        if used_symbol != symbol:
            st.info(f"🔁 تم استبدال {symbol} بـ {used_symbol}")
        result = {}
        t = threading.Thread(target=train_model, args=(df,result))
        t.start()
        t.join()  # انتظر التدريب فقط
        model = result.get("model")
        acc = result.get("acc")
        if model is None:
            st.error("⚠ النموذج لم يتم تدريبه بسبب قلة البيانات")
            return
        pred = predict_last(model, df)
        if pred==1:
            st.success(f"🔥 التنبؤ: {symbol} صاعد (شراء تعليمي)")
        else:
            st.warning(f"📉 التنبؤ: {symbol} هابط (تجنب الشراء)")
        st.success(f"✔ دقة النموذج: {acc*100:.2f}%")
        st.dataframe(df.tail(5))
        if uploaded_file:
            img_pred = analyze_image(uploaded_file)
            if img_pred==1: st.success("🔥 الصورة تشير للسوق صاعد")
            elif img_pred==0: st.warning("📉 الصورة تشير للسوق هابط")
            else: st.info("⚠ لم يتمكن التطبيق من تحليل الصورة")

if st.button("📊 تحديث الآن"):
    update()

# تحديث تلقائي كل X ثواني
st_autorefresh = st.empty()
st_autorefresh.info(f"⏱ التحديث التلقائي كل {update_sec} ثانية")
st.experimental_rerun()  # يمكن استبداله بتقنية timer في نسخ متقدمة