# app.py
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

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader — النسخة النهائية 💜", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols

FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "Price_Range", "Price_Change", "MA_5", "Volume_MA"]

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر سهم أو زوج الفوركس:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2023,1,1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
uploaded_file = st.sidebar.file_uploader("ارفع صورة الشموع/المنحنيات", type=["png","jpg","jpeg"])

# إعادة المحاولة
if st.sidebar.button("🔄 إعادة المحاولة"):
    if hasattr(st, "runtime"):
        st.runtime.legacy_caching.clear_cache()
    st.experimental_rerun()

# التحقق من التواريخ
if start_date >= end_date:
    st.sidebar.error("⚠ تاريخ البداية يجب أن يكون قبل تاريخ النهاية")
    st.stop()
if start_date > datetime.date.today():
    st.sidebar.error("⚠ تاريخ البداية لا يمكن أن يكون في المستقبل")
    st.stop()

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

# ===== تدريب النموذج =====
def train_model(df):
    X, y, df_feat = prepare_features(df, with_target=True)
    if X is None or y is None or len(X) < 30:
        st.warning("⚠ البيانات غير كافية لتدريب النموذج بدقة.")
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
    except Exception as e:
        st.error(f"⚠ خطأ في تدريب النموذج: {e}")
        return None, None

# ===== التنبؤ بآخر صف =====
def predict_last(model, df):
    X_pred, _, _ = prepare_features(df, with_target=False)
    if X_pred is None or X_pred.empty:
        st.warning("⚠ لا توجد بيانات كافية للتنبؤ.")
        return None
    last_row = X_pred.iloc[[-1]].values
    try:
        return model.predict(last_row)[0]
    except Exception as e:
        st.error(f"⚠ خطأ أثناء التنبؤ: {e}")
        return None

# ===== تحليل الصور =====
def analyze_image(file):
    try:
        image = Image.open(file).convert("RGB")
        image = image.resize((256,256))
        st.image(image, caption="📊 الصورة المحملة", use_column_width=True)
        img_cv = np.array(image)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        mean_val = float(np.mean(img_gray))
        st.write(f"📊 متوسط الإضاءة: {mean_val:.1f}")
        return 1 if mean_val > 120 else 0
    except Exception as e:
        st.error(f"⚠ خطأ في تحليل الصورة: {e}")
        return None

# ===== واجهة التطبيق =====
st.title("📈 AI Smart Trader — النسخة النهائية 💜")
st.warning("⚠ التوصيات تعليمية فقط، التداول يحمل مخاطر مالية، استشر مختصاً قبل أي قرار.")

if st.button("📊 الحصول على التوصيات"):
    with st.spinner("⏳ جاري تحميل البيانات وتحليلها..."):
        df, used_symbol = load_data_with_fallback(symbol, start_date, end_date)
        if df.empty:
            st.error("⚠ لا توجد بيانات كافية لهذا الرمز أو البدائل.")
            st.stop()

        if used_symbol != symbol:
            st.info(f"🔁 تم استبدال {symbol} بـ {used_symbol} لعدم توفر بيانات كافية.")
            symbol = used_symbol

        model, acc = train_model(df)
        if model is None:
            st.error("⚠ لم يتم تدريب النموذج بسبب قلة البيانات.")
            st.stop()

        pred = predict_last(model, df)
        if pred is None:
            st.error("⚠ لم يتمكن النموذج من التنبؤ.")
            st.stop()

        st.success(f"✔ دقة النموذج على بيانات الاختبار: {acc*100:.2f}%")
        if pred == 1:
            st.success(f"🔥 التنبؤ: {symbol} صاعد (إشارة شراء تعليمية)")
        else:
            st.warning(f"📉 التنبؤ: {symbol} هابط أو ضعيف (تجنب الشراء)")

        st.markdown("### آخر البيانات التاريخية:")
        st.dataframe(df.tail(10))

        st.markdown("### 📈 إحصائيات أساسية")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("متوسط الإغلاق", f"{df['Close'].mean():.2f}")
        with col2: st.metric("أعلى سعر", f"{df['High'].max():.2f}")
        with col3: st.metric("أقل سعر", f"{df['Low'].min():.2f}")

        if uploaded_file is not None:
            st.markdown("### 📷 تحليل الصورة")
            img_pred = analyze_image(uploaded_file)
            if img_pred == 1: st.success("🔥 السوق صاعد")
            elif img_pred == 0: st.warning("📉 السوق هابط")
            else: st.info("⚠ لم يتمكن التطبيق من تحليل الصورة")

st.markdown("---")
st.subheader("⭐ رموز مقترحة للمراقبة (تعليميًا)")
st.write(random.sample(all_symbols, 5))
