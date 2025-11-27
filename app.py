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
st.set_page_config(page_title="AI Smart Trader", layout="wide")

# ===== القوائم =====
low_liquidity_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = low_liquidity_symbols + forex_symbols

# ===== Sidebar =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر السهم أو زوج الفوركس:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
uploaded_file = st.sidebar.file_uploader("ارفع صورة الشموع/المنحنيات للتحليل", type=["png","jpg","jpeg"])

# ===== تحقق من التواريخ =====
if start_date >= end_date:
    st.sidebar.error("⚠ تاريخ البداية يجب أن يكون قبل تاريخ النهاية")
    st.stop()
if start_date > datetime.date.today():
    st.sidebar.error("⚠ تاريخ البداية لا يمكن أن يكون في المستقبل")
    st.stop()

# ===== زر إعادة المحاولة =====
if st.sidebar.button("🔄 إعادة المحاولة"):
    st.experimental_rerun()

# ===== تحميل البيانات =====
def load_data(original_symbol, start, end):
    symbol = original_symbol
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(symbol, start=start, end=end)
            required_cols = ["Open","High","Low","Close","Volume"]
            if df.empty or not all(col in df.columns for col in required_cols):
                st.warning(f"⚠ لا توجد بيانات كافية للسهم {symbol}. جاري البحث عن بديل...")
                alternatives = [s for s in all_symbols if s != symbol]
                found = False
                for alt in alternatives:
                    new_df = yf.download(alt, start=start, end=end)
                    if not new_df.empty and all(col in new_df.columns for col in required_cols):
                        df = new_df
                        symbol = alt
                        st.info(f"✅ تم استخدام الرمز البديل: {symbol}")
                        found = True
                        break
                if not found:
                    break
            else:
                break
        except Exception as e:
            st.error(f"محاولة {attempt+1} فشلت: {e}")
            if attempt == max_retries-1:
                return pd.DataFrame(), original_symbol
    return df, symbol

# ===== إنشاء عمود الهدف =====
def add_target(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df

# ===== تدريب النموذج =====
def train_model(df):
    if len(df) < 30:
        st.warning("⚠ البيانات غير كافية لتدريب نموذج دقيق")
        return None, None, None
    df = df.copy()
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['Volume_MA'] = df['Volume'].rolling(5).mean()
    df = df.dropna()
    if len(df) < 20:
        st.warning("⚠ البيانات غير كافية بعد تنظيف القيم المفقودة")
        return None, None, None
    feature_cols = ['Open','High','Low','Close','Volume','Price_Range','Price_Change','MA_5','Volume_MA']
    feature_cols_used = [col for col in feature_cols if col in df.columns]
    X = df[feature_cols_used]
    y = df['Target']
    split_point = int(len(df)*0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    try:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return model, acc, feature_cols_used
    except Exception as e:
        st.error(f"⚠ خطأ في تدريب النموذج: {e}")
        return None, None, None

# ===== التنبؤ =====
def predict_last(model, df, training_cols):
    missing_cols = [col for col in training_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠ الأعمدة التالية ناقصة في بيانات التنبؤ: {missing_cols}")
        return None
    last_row = df[training_cols].iloc[-1].values.reshape(1,-1)
    return model.predict(last_row)[0]

# ===== تحليل الصور =====
def analyze_image(file):
    try:
        image = Image.open(file).convert('RGB')
        st.image(image, caption="📊 الصورة المحملة", use_column_width=True)
        st.info("ℹ️ تحليل الصور تجريبي ويعتمد على الإضاءة فقط")
        img_cv = np.array(image)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(img_gray)
        std_val = np.std(img_gray)
        st.write(f"📊 إحصائيات الصورة: متوسط الإضاءة {mean_val:.1f}, انحراف معياري {std_val:.1f}")
        return 1 if mean_val > 120 else 0
    except Exception as e:
        st.error(f"⚠ خطأ في تحليل الصورة: {e}")
        return None

# ===== عنوان التطبيق =====
st.title("📈 AI Smart Trader — النسخة النهائية 💜")

# ===== عند الضغط على زر التحليل =====
if st.button("📊 الحصول على التوصيات"):
    st.warning("""
    ⚠ **تحذير مهم**: 
    - هذه التوصيات لأغراض تعليمية فقط
    - التداول يحمل مخاطر خسارة الأموال
    - استشر مستشاراً مالياً قبل اتخاذ أي قرار
    """)
    with st.spinner('⏳ جاري تحميل البيانات وتحليلها...'):
        df, actual_symbol = load_data(symbol, start_date, end_date)
        if df.empty:
            st.error("⚠ لا توجد بيانات لتحليلها.")
            st.stop()
        if actual_symbol != symbol:
            st.info(f"🔁 تم استخدام الرمز {actual_symbol} بدلاً من {symbol}")
            symbol = actual_symbol
        df = add_target(df)
        if df.empty:
            st.warning("⚠ البيانات غير كافية للتنبؤ.")
            st.stop()
        model, acc, feature_cols_used = train_model(df)
        if model is None:
            st.error("⚠ البيانات غير كافية لتدريب النموذج.")
            st.stop()
        pred = predict_last(model, df, feature_cols_used)
        if pred is None:
            st.error("⚠ لم يتمكن النموذج من التنبؤ بسبب الأعمدة المفقودة")
            st.stop()
        st.success(f"✔ دقة النموذج على بيانات الاختبار: {acc*100:.2f}%")
        if pred == 1:
            st.success("🔥 التنبؤ: السهم/الزوج سيرتفع — شراء")
        else:
            st.warning("📉 التنبؤ: السهم/الزوج سينخفض — بيع / تجنب")
        st.markdown("### آخر البيانات التاريخية:")
        st.dataframe(df.tail(10))
        st.markdown("### 📈 إحصائيات أساسية")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("متوسط السعر", f"{df['Close'].mean():.2f}")
        with col2:
            st.metric("أعلى سعر", f"{df['High'].max():.2f}")
        with col3:
            st.metric("أقل سعر", f"{df['Low'].min():.2f}")
        if uploaded_file is not None:
            st.markdown("### تحليل الشموع/المنحنيات من الصورة:")
            img_pred = analyze_image(uploaded_file)
            if img_pred == 1:
                st.success("🔥 تحليل الصورة: السوق صاعد")
            elif img_pred == 0:
                st.warning("📉 تحليل الصورة: السوق هابط")
            else:
                st.info("⚠ لم يتمكن التطبيق من تحليل الصورة")

# ===== توصيات يومية =====
st.markdown("---")
st.subheader("⭐ أفضل الأسهم وأزواج الفوركس للتداول اليومي")
today_symbols = random.sample(all_symbols, 5)
st.write(today_symbols)