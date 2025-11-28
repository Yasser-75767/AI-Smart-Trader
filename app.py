# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from PIL import Image
import cv2
import datetime
import random

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader — النسخة المستقرة 💜", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols

# الأعمدة المستخدمة
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Price_Range", "Price_Change", "MA_5", "Volume_MA",
    "RSI", "MACD", "MA20", "BB_upper", "BB_lower"
]

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر سهم أو زوج الفوركس:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2023,1,1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
uploaded_file = st.sidebar.file_uploader("ارفع صورة الشموع/المنحنيات للتحليل", type=["png","jpg","jpeg"])

# ===== دوال التطبيق =====

# تحميل البيانات مع بديل
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty or len(df) < 10:
            for alt in [s for s in all_symbols if s != symbol]:
                try:
                    df_alt = yf.download(alt, start=start, end=end, progress=False)
                    if not df_alt.empty and len(df_alt) >= 10:
                        st.info(f"ℹ تم استخدام الرمز البديل: {alt}")
                        return df_alt, alt
                except:
                    continue
            return pd.DataFrame(), symbol
        return df, symbol
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {e}")
        return pd.DataFrame(), symbol

# إضافة مؤشرات فنية
def add_technical_indicators(df):
    try:
        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        df['RSI'] = 100 - (100/(1+rs))
        # MACD
        exp1 = df['Close'].ewm(span=12, min_periods=1).mean()
        exp2 = df['Close'].ewm(span=26, min_periods=1).mean()
        df['MACD'] = exp1 - exp2
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
        df['BB_std'] = df['Close'].rolling(20, min_periods=1).std()
        df['BB_upper'] = df['MA20'] + (df['BB_std']*2)
        df['BB_lower'] = df['MA20'] - (df['BB_std']*2)
        return df
    except Exception as e:
        st.error(f"خطأ في المؤشرات الفنية: {e}")
        return df

# تجهيز الميزات
def prepare_features(df, with_target=True):
    if df.empty: return None, None, None
    df = df.copy()
    required_cols = ["Open","High","Low","Close","Volume"]
    if not all(c in df.columns for c in required_cols): return None, None, None
    df["Price_Range"] = df["High"] - df["Low"]
    df["Price_Change"] = df["Close"] - df["Open"]
    df["MA_5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["Volume_MA"] = df["Volume"].rolling(5, min_periods=1).mean()
    df = add_technical_indicators(df)
    for col in FEATURE_COLS:
        if col not in df.columns: df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    if with_target:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna(subset=["Target"])
        if df.empty: return None, None, None
        X = df[FEATURE_COLS]
        y = df["Target"].astype(int)
        return X, y, df
    else:
        X = df[FEATURE_COLS]
        return X, df, None

# تدريب النموذج
def train_model(df):
    X, y, _ = prepare_features(df, with_target=True)
    if X is None or y is None or len(X)<30:
        st.warning("⚠ البيانات غير كافية لتدريب النموذج (30 نقطة على الأقل)")
        return None, None
    split = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    try:
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            tree_method="hist", use_label_encoder=False,
            eval_metric="logloss", random_state=42
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return model, acc
    except Exception as e:
        st.error(f"⚠ خطأ في تدريب النموذج: {e}")
        return None, None

# التنبؤ
def predict_last(model, df):
    X_pred, _, _ = prepare_features(df, with_target=False)
    if X_pred is None or X_pred.empty: return None
    try:
        last_row = X_pred.iloc[[-1]].values
        return model.predict(last_row)[0]
    except Exception as e:
        st.error(f"⚠ خطأ أثناء التنبؤ: {e}")
        return None

# تحليل الصور
def analyze_image(file):
    try:
        image = Image.open(file).convert("RGB")
        image = image.resize((256,256))
        st.image(image, caption="📊 الصورة المحملة", use_column_width=True)
        img_cv = np.array(image)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        mean_val = float(np.mean(img_gray))
        st.write(f"📊 متوسط الإضاءة في الصورة: {mean_val:.1f}")
        return 1 if mean_val>120 else 0
    except Exception as e:
        st.error(f"⚠ خطأ في تحليل الصورة: {e}")
        return None

# ===== واجهة التطبيق =====
st.title("📈 AI Smart Trader — النسخة المستقرة 💜")
st.warning("⚠ التوصيات تعليمية فقط، التداول يحمل مخاطر مالية")

if st.button("📊 الحصول على التوصيات"):
    with st.spinner("⏳ جاري تحميل البيانات وتحليلها..."):
        df, used_symbol = load_data(symbol, start_date, end_date)
        if df.empty or len(df)<10:
            st.error("⚠ لا توجد بيانات كافية لهذا الرمز أو البدائل")
            st.stop()
        st.info(f"📊 تم تحميل {len(df)} يوم تداول للرمز {used_symbol}")
        model, acc = train_model(df)
        if model is None:
            st.error("⚠ لم يتمكن النموذج من التدريب بسبب قلة البيانات")
            st.stop()
        pred = predict_last(model, df)
        if pred is None:
            st.error("⚠ لا يمكن التنبؤ بالاتجاه حالياً")
        else:
            st.success(f"✔ دقة النموذج: {acc*100:.2f}%")
            if pred==1:
                st.success(f"🔥 التنبؤ: {used_symbol} صاعد (إشارة شراء تعليمية)")
            else:
                st.warning(f"📉 التنبؤ: {used_symbol} هابط (تجنب الشراء)")
        st.markdown("### آخر البيانات التاريخية:")
        st.dataframe(df.tail(10))
        if uploaded_file:
            st.markdown("### 📷 تحليل الصورة")
            img_pred = analyze_image(uploaded_file)
            if img_pred==1: st.success("🔥 تحليل الصورة: السوق يبدو صاعداً")
            elif img_pred==0: st.warning("📉 تحليل الصورة: السوق يبدو هابطاً")
            else: st.info("⚠ لم يتمكن التطبيق من تحليل الصورة")

st.markdown("---")
st.subheader("⭐ رموز مقترحة للمراقبة (تعليمي)")
st.write(random.sample(all_symbols,5))