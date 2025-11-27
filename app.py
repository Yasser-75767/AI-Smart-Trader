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
st.set_page_config(page_title="AI Smart Trader — النسخة النهائية", layout="wide")

# ===== الرموز =====
low_liquidity_symbols = ["AAPL","MSFT","GOOGL","NVDA","AMZN"]
forex_symbols = ["EURUSD=X","USDJPY=X","GBPUSD=X","USDCHF=X","AUDUSD=X"]
all_symbols = low_liquidity_symbols + forex_symbols

# ===== Sidebar =====
st.sidebar.header("إعدادات التطبيق")

symbol = st.sidebar.selectbox("اختر السهم أو زوج الفوركس:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2023,1,1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
uploaded_file = st.sidebar.file_uploader("ارفع صورة الشموع/المنحنيات للتحليل", type=["png","jpg","jpeg"])

# تحقق من صحة التواريخ
if start_date >= end_date:
    st.sidebar.error("❌ تاريخ البداية يجب أن يكون قبل تاريخ النهاية")
    st.stop()

if start_date > datetime.date.today():
    st.sidebar.error("❌ تاريخ البداية لا يمكن أن يكون في المستقبل")
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
            df = yf.download(symbol, start=start, end=end, progress=False)
            required_cols = ["Open","High","Low","Close","Volume"]
            if df.empty or not all(col in df.columns for col in required_cols):
                st.warning(f"⚠ لا توجد بيانات كافية للسهم {symbol}. البحث عن بديل...")
                alternatives = [s for s in all_symbols if s != symbol]
                for alt in alternatives:
                    new_df = yf.download(alt, start=start, end=end, progress=False)
                    if not new_df.empty and all(col in new_df.columns for col in required_cols):
                        df = new_df
                        symbol = alt
                        st.info(f"✅ تم استخدام الرمز البديل: {symbol}")
                        break
            # إضافة الأعمدة المحسوبة فورًا
            if not df.empty:
                df['Price_Range'] = df['High'] - df['Low']
                df['Price_Change'] = df['Close'] - df['Open']
            break
        except Exception as e:
            st.error(f"محاولة {attempt+1} فشلت: {e}")
            if attempt == max_retries-1:
                return pd.DataFrame(), original_symbol
    return df, symbol

# ===== إضافة عمود الهدف =====
def add_target(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df

# ===== تدريب النموذج =====
def train_model(df):
    if len(df) < 30:
        st.warning("⚠ البيانات غير كافية لتدريب نموذج دقيق (تحتاج 30 نقطة على الأقل)")
        return None, None

    df = df.copy()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
    df = df.dropna()

    feature_cols = ['Open','High','Low','Close','Volume','Price_Range','Price_Change']
    available_cols = [col for col in feature_cols + ['MA_5','Volume_MA'] if col in df.columns]

    X = df[available_cols]
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
        return model, acc
    except Exception as e:
        st.error(f"⚠ خطأ في تدريب النموذج: {e}")
        return None, None

# ===== التنبؤ بالبيانات الأخيرة =====
def predict_last(model, df):
    base_cols = ['Open','High','Low','Close','Volume']
    feature_cols = base_cols.copy()
    if 'Price_Range' in df.columns: feature_cols.append('Price_Range')
    if 'Price_Change' in df.columns: feature_cols.append('Price_Change')
    if 'MA_5' in df.columns: feature_cols.append('MA_5')
    if 'Volume_MA' in df.columns: feature_cols.append('Volume_MA')
    last_row = df[feature_cols].iloc[-1].values.reshape(1,-1)
    return model.predict(last_row)[0]

# ===== تحليل الصور =====
def analyze_image(file):
    try:
        image = Image.open(file).convert('RGB')
        st.image(image, caption="📊 الصورة المحملة", use_column_width=True)
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
st.markdown("""
⚠ **تحذير مهم**:  
- هذه التوصيات لأغراض تعليمية فقط  
- التداول يحمل مخاطر خسارة الأموال  
- استشر مستشاراً مالياً قبل اتخاذ أي قرار
""")

# ===== زر الحصول على التوصيات =====
if st.button("📊 الحصول على التوصيات"):
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

        model, acc = train_model(df)
        if model is None:
            st.error("⚠ النموذج لم يتمكن من التدريب بسبب نقص البيانات")
            st.stop()

        pred = predict_last(model, df)
        st.success(f"✔ دقة النموذج على بيانات الاختبار: {acc*100:.2f}%")
        if pred == 1:
            st.success("🔥 التنبؤ: السهم/الزوج سيرتفع — شراء")
        else:
            st.warning("📉 التنبؤ: السهم/الزوج سينخفض — بيع / تجنب")

        st.markdown("### آخر البيانات التاريخية:")
        st.dataframe(df.tail(10))

        st.markdown("### 📈 إحصائيات أساسية")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("متوسط السعر", f"{df['Close'].mean():.2f}")
        with col2: st.metric("أعلى سعر", f"{df['High'].max():.2f}")
        with col3: st.metric("أقل سعر", f"{df['Low'].min():.2f}")

        if uploaded_file is not None:
            st.markdown("### تحليل الشموع/المنحنيات من الصورة:")
            img_pred = analyze_image(uploaded_file)
            if img_pred == 1:
                st.success("🔥 تحليل الصورة: السوق صاعد")
            elif img_pred == 0:
                st.warning("📉 تحليل الصورة: السوق هابط")
            else:
                st.info("⚠ لم يتمكن التطبيق من تحليل الصورة")

# ===== توصيات يومية عشوائية =====
st.markdown("---")
st.subheader("⭐ أفضل الأسهم وأزواج الفوركس للتداول اليومي")
today_symbols = random.sample(all_symbols, min(5,len(all_symbols)))
st.write(today_symbols)