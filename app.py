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
    "Price_Range", "Price_Change", "MA_5", "Volume_MA"
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

# تجهيز الميزات - الإصلاح الكامل
def prepare_features(df, with_target=True):
    if df.empty: 
        return None, None, None
    
    df = df.copy()
    
    # التحقق من الأعمدة الأساسية
    required_cols = ["Open","High","Low","Close","Volume"]
    if not all(col in df.columns for col in required_cols):
        return None, None, None
    
    try:
        # حساب الميزات الأساسية فقط
        df["Price_Range"] = df["High"] - df["Low"]
        df["Price_Change"] = df["Close"] - df["Open"]
        df["MA_5"] = df["Close"].rolling(5, min_periods=1).mean()
        df["Volume_MA"] = df["Volume"].rolling(5, min_periods=1).mean()
        
        # ملء القيم الناقصة
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
        
        if with_target:
            # إنشاء الهدف بشكل آمن
            df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
            
            # حذف الصف الأخير فقط (الذي يحتوي على NaN في Target)
            df_with_target = df.iloc[:-1].copy()
            
            if df_with_target.empty:
                return None, None, None
                
            X = df_with_target[FEATURE_COLS]
            y = df_with_target["Target"].astype(int)
            return X, y, df_with_target
        else:
            # للتنبؤ، نستخدم كل البيانات بما في ذلك الصف الأخير
            X = df[FEATURE_COLS]
            return X, df, None
            
    except Exception as e:
        st.error(f"⚠ خطأ في تجهيز الميزات: {str(e)}")
        return None, None, None

# تدريب النموذج
def train_model(df):
    try:
        X, y, df_processed = prepare_features(df, with_target=True)
        
        if X is None or y is None:
            st.warning("⚠ لا توجد بيانات كافية لتدريب النموذج")
            return None, None
            
        if len(X) < 30:
            st.warning(f"⚠ البيانات غير كافية لتدريب النموذج ({len(X)} نقطة فقط، تحتاج 30 على الأقل)")
            return None, None
        
        split = int(len(X) * 0.8)
        if split == 0:
            st.warning("⚠ لا توجد بيانات كافية للتقسيم")
            return None, None
            
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        if len(X_test) == 0:
            st.warning("⚠ بيانات الاختبار فارغة")
            return None, None
            
        model = xgb.XGBClassifier(
            n_estimators=100, 
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
        st.error(f"⚠ خطأ في تدريب النموذج: {str(e)}")
        return None, None

# التنبؤ
def predict_last(model, df):
    try:
        X_pred, df_processed, _ = prepare_features(df, with_target=False)
        if X_pred is None or X_pred.empty: 
            return None
        
        last_row = X_pred.iloc[[-1]].values
        prediction = model.predict(last_row)[0]
        return prediction
        
    except Exception as e:
        st.error(f"⚠ خطأ أثناء التنبؤ: {str(e)}")
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
        return 1 if mean_val > 120 else 0
    except Exception as e:
        st.error(f"⚠ خطأ في تحليل الصورة: {str(e)}")
        return None

# ===== واجهة التطبيق =====
st.title("📈 AI Smart Trader — النسخة المستقرة 💜")
st.warning("⚠ التوصيات تعليمية فقط، التداول يحمل مخاطر مالية")

if st.button("📊 الحصول على التوصيات"):
    with st.spinner("⏳ جاري تحميل البيانات وتحليلها..."):
        try:
            # تحميل البيانات
            df, used_symbol = load_data(symbol, start_date, end_date)
            if df.empty or len(df) < 10:
                st.error("⚠ لا توجد بيانات كافية لهذا الرمز أو البدائل")
                st.stop()
            
            st.success(f"📊 تم تحميل {len(df)} يوم تداول للرمز {used_symbol}")
            
            # عرض معلومات أساسية عن البيانات
            st.write("### 📈 معلومات أساسية عن البيانات:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("متوسط الإغلاق", f"{df['Close'].mean():.2f}")
            with col2:
                st.metric("أعلى سعر", f"{df['High'].max():.2f}")
            with col3:
                st.metric("أقل سعر", f"{df['Low'].min():.2f}")
            
            # تدريب النموذج
            model, acc = train_model(df)
            if model is None:
                st.error("⚠ لم يتمكن النموذج من التدريب بسبب مشاكل في البيانات")
                st.stop()
            
            # التنبؤ
            pred = predict_last(model, df)
            if pred is None:
                st.error("⚠ لا يمكن التنبؤ بالاتجاه حالياً")
            else:
                st.success(f"✔ دقة النموذج: {acc*100:.2f}%")
                if pred == 1:
                    st.success(f"🎯 التنبؤ: {used_symbol} صاعد (اتجاه إيجابي)")
                    st.info("💡 الإشارة: قد تكون فرصة للشراء (تعليمي)")
                else:
                    st.warning(f"📉 التنبؤ: {used_symbol} هابط (اتجاه سلبي)")
                    st.info("💡 الإشارة: قد تكون فرصة للبيع أو الانتظار (تعليمي)")
            
            # عرض البيانات
            st.markdown("### 📊 آخر البيانات التاريخية:")
            st.dataframe(df.tail(10))
            
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
                    
        except Exception as e:
            st.error(f"⚠ حدث خطأ غير متوقع: {str(e)}")
            st.stop()

st.markdown("---")
st.subheader("⭐ رموز مقترحة للمراقبة")
recommended_symbols = random.sample(all_symbols, min(3, len(all_symbols)))
st.write(recommended_symbols)

st.markdown("---")
st.info("""
### 📝 ملاحظات مهمة:
- هذا التطبيق لأغراض تعليمية فقط
- الدقة التنبؤية قد تختلف حسب ظروف السوق
- استشر خبراء ماليين قبل اتخاذ أي قرارات تداول حقيقية
""")