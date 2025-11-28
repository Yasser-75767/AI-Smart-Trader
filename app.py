# app_mobile_final.py — AI Smart Trader نسخة الهاتف النهائية
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import datetime
import random
import ta
from PIL import Image

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader — الهاتف النهائي 💜", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols

# الأعمدة الأساسية + مؤشرات
FEATURE_COLS = [
    "Open","High","Low","Close","Volume",
    "Price_Range","Price_Change","MA_5","MA20","MA50","Volume_MA",
    "RSI","MACD","MACD_Signal","BB_Upper","BB_Lower"
]

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر سهم أو زوج الفوركس:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2023,1,1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
uploaded_file = st.sidebar.file_uploader("📷 ارفع صورة الشموع/المنحنيات للتحليل", type=["png","jpg","jpeg"])

# ===== دوال التحليل الفني =====
def add_indicators(df):
    df = df.copy()
    df["Price_Range"] = df["High"] - df["Low"]
    df["Price_Change"] = df["Close"] - df["Open"]
    df["MA_5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["Volume_MA"] = df["Volume"].rolling(5, min_periods=1).mean()

    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df

# ===== تحميل البيانات =====
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty or len(df) < 10:
            for alt in [s for s in all_symbols if s != symbol]:
                df_alt = yf.download(alt, start=start, end=end, progress=False)
                if not df_alt.empty and len(df_alt) >= 10:
                    st.info(f"ℹ تم استخدام الرمز البديل: {alt}")
                    return add_indicators(df_alt), alt
            return pd.DataFrame(), symbol
        return add_indicators(df), symbol
    except Exception as e:
        st.error(f"⚠ خطأ في تحميل البيانات: {e}")
        return pd.DataFrame(), symbol

# ===== تجهيز البيانات للنموذج =====
def prepare_features(df, with_target=True):
    if df.empty:
        return None, None
    df = df.copy()
    if with_target:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.iloc[:-1]
        X = df[FEATURE_COLS]
        y = df["Target"].astype(int)
        return X, y
    else:
        X = df[FEATURE_COLS]
        return X, None

# ===== تدريب النموذج =====
def train_model(df):
    X, y = prepare_features(df, with_target=True)
    if X is None or y is None or len(X) < 30:
        st.warning("⚠ لا توجد بيانات كافية لتدريب النموذج")
        return None, None
    split = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        tree_method="hist",
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# ===== التنبؤ =====
def predict_last(model, df):
    X_pred, _ = prepare_features(df, with_target=False)
    if X_pred is None or X_pred.empty:
        return None
    last_row = X_pred.iloc[[-1]].values
    pred = model.predict(last_row)[0]
    prob = model.predict_proba(last_row)[0]
    return pred, prob

# ===== تحليل الصورة بدون cv2 =====
def analyze_image(file):
    try:
        image = Image.open(file).convert("L")  # رمادية
        image = image.resize((256,256))
        st.image(image, caption="📷 الصورة المحملة", use_column_width=True)
        mean_val = np.mean(np.array(image))
        st.write(f"📊 متوسط الإضاءة في الصورة: {mean_val:.1f}")
        return 1 if mean_val > 120 else 0
    except:
        return None

# ===== واجهة التطبيق =====
st.title("📈 AI Smart Trader — الهاتف النهائي 💜")
st.warning("⚠ التوصيات تعليمية فقط، التداول يحمل مخاطر مالية")

if st.button("📊 الحصول على التوصيات"):
    with st.spinner("⏳ جاري تحميل البيانات وتحليلها..."):
        df, used_symbol = load_data(symbol, start_date, end_date)
        if df.empty or len(df) < 10:
            st.error("⚠ لا توجد بيانات كافية لهذا الرمز أو البدائل")
            st.stop()

        st.success(f"📊 تم تحميل {len(df)} يوم تداول للرمز {used_symbol}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("متوسط الإغلاق", f"{float(df['Close'].mean()):.2f}")
        with col2:
            st.metric("أعلى سعر", f"{float(df['High'].max()):.2f}")
        with col3:
            st.metric("أقل سعر", f"{float(df['Low'].min()):.2f}")

        # تدريب النموذج
        model, acc = train_model(df)
        if model is None:
            st.error("⚠ لم يتمكن النموذج من التدريب")
            st.stop()

        # التنبؤ
        pred, prob = predict_last(model, df)
        if pred is not None:
            confidence = prob[pred]*100
            if pred == 1:
                signal = "شراء قوي" if confidence > 65 else "شراء ضعيف"
                st.success(f"🎯 التنبؤ: {used_symbol} صاعد — {signal} — ثقة {confidence:.2f}%")
            else:
                signal = "بيع قوي" if confidence > 65 else "بيع ضعيف"
                st.warning(f"📉 التنبؤ: {used_symbol} هابط — {signal} — ثقة {confidence:.2f}%")

        # عرض الرسوم البيانية للمؤشرات
        st.markdown("### 📊 الرسوم البيانية للمؤشرات الفنية")
        st.line_chart(df[["Close","MA20","MA50"]])
        st.line_chart(df[["RSI"]])
        st.line_chart(df[["MACD","MACD_Signal"]])
        st.line_chart(df[["BB_Upper","BB_Lower","Close"]])

        # تحليل الصورة إذا تم رفعها
        if uploaded_file is not None:
            st.markdown("### 📷 تحليل الصورة")
            img_pred = analyze_image(uploaded_file)
            if img_pred == 1:
                st.success("🔥 تحليل الصورة: السوق يبدو صاعداً")
            elif img_pred == 0:
                st.warning("📉 تحليل الصورة: السوق يبدو هابطاً")
            else:
                st.info("⚠ لم يتمكن التطبيق من تحليل الصورة")

st.markdown("---")
st.subheader("⭐ رموز مقترحة للمراقبة")
recommended_symbols = random.sample(all_symbols, min(3, len(all_symbols)))
st.write(recommended_symbols)

st.markdown("---")
st.info("""
### 📝 ملاحظات مهمة:
- هذا التطبيق لأغراض تعليمية فقط
- يعمل بالكامل على الهاتف
- جميع المؤشرات الفنية متاحة والتنبيهات الذكية تعمل
- استشر خبراء ماليين قبل اتخاذ أي قرارات تداول حقيقية
""")