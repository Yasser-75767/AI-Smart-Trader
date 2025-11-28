# app_final.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageFilter, ImageStat
import datetime
import random
import ta

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader — النسخة الدقيقة جداً 💎", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META", "NFLX"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر السهم أو الزوج:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())

confidence_threshold = st.sidebar.slider("حد الثقة (%)", 50, 95, 80)
uploaded_file = st.sidebar.file_uploader("رفع صورة الشموع/المنحنيات للتحليل", type=["png","jpg","jpeg"])

# ===== دوال محسنة بدون cv2 =====

def load_data(symbol, start, end):
    """تحميل بيانات محسنة"""
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        if data.empty or len(data) < 100:
            st.error("❌ البيانات غير كافية. تحتاج 100 يوم على الأقل")
            return pd.DataFrame(), symbol
        return data, symbol
    except Exception as e:
        st.error(f"❌ خطأ في التحميل: {e}")
        return pd.DataFrame(), symbol

def calculate_advanced_indicators(data):
    """حساب مؤشرات فنية متقدمة"""
    data = data.copy()
    
    # المتوسطات المتحركة
    for period in [5, 10, 20, 50]:
        data[f'MA_{period}'] = data['Close'].rolling(period).mean()
    
    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_Upper'] = bollinger.bollinger_hband()
    data['BB_Lower'] = bollinger.bollinger_lband()
    
    # مؤشرات الحجم
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    
    # التقلب
    data['Volatility'] = data['Close'].pct_change().rolling(20).std()
    
    # أنماط السعر
    data['Price_Range'] = data['High'] - data['Low']
    data['Price_Change'] = data['Close'] - data['Open']
    data['Gap'] = data['Open'] - data['Close'].shift(1)
    
    return data

def prepare_advanced_features(data, with_target=True):
    """تحضير الميزات المتقدمة"""
    if data.empty or len(data) < 50:
        return None, None, None
    
    try:
        data = calculate_advanced_indicators(data)
        
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Volatility',
            'Price_Range', 'Price_Change', 'Gap'
        ]
        
        # ملء القيم الناقصة
        data = data.fillna(method='ffill').fillna(0)
        
        if with_target:
            data["Target"] = (data['Close'].shift(-1) > data['Close']).astype(int)
            clean_data = data.iloc[:-1].copy()
            
            if clean_data.empty:
                return None, None, None
                
            X = clean_data[features]
            y = clean_data["Target"]
            return X, y, clean_data
        else:
            X = data[features]
            return X, data, None
            
    except Exception as e:
        st.error(f"⚠ خطأ في تجهيز الميزات: {str(e)}")
        return None, None, None

def train_advanced_model(data):
    """تدريب نموذج متقدم"""
    X, y, processed_data = prepare_advanced_features(data, with_target=True)
    
    if X is None or y is None or len(X) < 100:
        st.warning("⚠ تحتاج إلى 100 نقطة بيانات على الأقل للتدريب المتقدم")
        return None, None, None
    
    try:
        # تقسيم البيانات الزمني
        tscv = TimeSeriesSplit(n_splits=5)
        scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            tree_method="hist",
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        
        # تدريب مع تحقق متقاطع
        accuracy_scores = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            accuracy_scores.append(accuracy_score(y_test, model.predict(X_test)))
        
        avg_accuracy = np.mean(accuracy_scores)
        
        # التدريب النهائي على كل البيانات
        model.fit(X_scaled, y)
        
        return model, avg_accuracy, scaler
        
    except Exception as e:
        st.error(f"⚠ خطأ في التدريب: {str(e)}")
        return None, None, None

def predict_with_confidence(model, scaler, data):
    """التنبؤ مع حساب درجة الثقة"""
    X_pred, processed_data, _ = prepare_advanced_features(data, with_target=False)
    
    if X_pred is None or X_pred.empty:
        return None, None
    
    try:
        X_scaled = scaler.transform(X_pred)
        
        prediction = model.predict(X_scaled[-1:])[0]
        probabilities = model.predict_proba(X_scaled[-1:])[0]
        
        confidence = max(probabilities) * 100
        
        return prediction, confidence
        
    except Exception as e:
        st.error(f"⚠ خطأ في التنبؤ: {str(e)}")
        return None, None

def analyze_image_advanced(file):
    """تحليل متقدم للصور بدون cv2"""
    try:
        image = Image.open(file).convert("RGB")
        image = image.resize((400, 400))
        
        st.image(image, caption="📊 الصورة المحملة", use_column_width=False, width=300)
        
        # تحليل الصورة باستخدام PIL فقط
        # تحويل إلى رمادي
        gray_image = image.convert('L')
        
        # إحصائيات الصورة
        stat = ImageStat.Stat(gray_image)
        mean_brightness = stat.mean[0]
        std_brightness = stat.stddev[0]
        
        # تحليل الحواف باستخدام مرشح PIL
        edges = image.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges.convert('L'))
        edge_intensity = edge_stat.mean[0]
        
        # تحليل التباين
        contrast = image.filter(ImageFilter.CONTOUR)
        contrast_stat = ImageStat.Stat(contrast.convert('L'))
        contrast_level = contrast_stat.mean[0]
        
        # نظام تسجيل متقدم
        score = 0
        if mean_brightness > 130: score += 1
        if edge_intensity > 30: score += 1
        if std_brightness > 40: score += 1
        if contrast_level > 50: score += 1
        
        st.write("**📈 تحليل الصورة المتقدم:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("متوسط الإضاءة", f"{mean_brightness:.1f}")
        with col2:
            st.metric("شدة الحواف", f"{edge_intensity:.1f}")
        with col3:
            st.metric("التباين", f"{std_brightness:.1f}")
        with col4:
            st.metric("التفاصيل", f"{contrast_level:.1f}")
        
        return 1 if score >= 2 else 0, score
        
    except Exception as e:
        st.error(f"⚠ خطأ في تحليل الصورة: {str(e)}")
        return None, 0

# ===== واجهة التطبيق =====
st.title("🎯 AI Smart Trader — النسخة الدقيقة جداً 💎")
st.warning("⚠ **تحذير:** هذه أداة تعليمية. التداول الفعلي يحمل مخاطر.")

if st.button("🚀 بدء التحليل الدقيق"):
    with st.spinner("🔬 جاري التحليل المتعمق..."):
        try:
            # تحميل البيانات
            data, used_symbol = load_data(symbol, start_date, end_date)
            if data.empty:
                st.stop()
            
            st.success(f"✅ تم تحميل {len(data)} يوم تداول لـ {used_symbol}")
            
            # عرض إحصائيات
            st.write("### 📊 الإحصائيات الأساسية:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("متوسط الإغلاق", f"{data['Close'].mean():.2f}")
            with col2:
                st.metric("أعلى سعر", f"{data['High'].max():.2f}")
            with col3:
                st.metric("أقل سعر", f"{data['Low'].min():.2f}")
            
            # مؤشرات فنية حالية
            st.write("### 📈 المؤشرات الفنية الحالية:")
            col4, col5, col6 = st.columns(3)
            
            with col4:
                current_rsi = ta.momentum.RSIIndicator(data['Close']).rsi().iloc[-1]
                st.metric("RSI", f"{current_rsi:.1f}")
            with col5:
                current_price = data['Close'].iloc[-1]
                ma_50 = data['Close'].rolling(50).mean().iloc[-1]
                trend = "📈 فوق" if current_price > ma_50 else "📉 تحت"
                st.metric("الاتجاه vs المتوسط 50", trend)
            with col6:
                volatility = data['Close'].pct_change().std() * 100
                st.metric("التقلب", f"{volatility:.2f}%")
            
            # تدريب النموذج
            model, accuracy, scaler = train_advanced_model(data)
            
            if model is None:
                st.error("❌ فشل في تدريب النموذج")
                st.stop()
            
            # التنبؤ
            prediction, confidence = predict_with_confidence(model, scaler, data)
            
            if prediction is not None:
                st.write("### 🎯 نتائج التحليل الدقيق:")
                
                # عرض النتائج
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    if prediction == 1:
                        st.success("**الاتجاه: 📈 صاعد**")
                        st.progress(0.8)
                    else:
                        st.error("**الاتجاه: 📉 هابط**")
                        st.progress(0.2)
                
                with result_col2:
                    if confidence >= confidence_threshold:
                        st.success(f"**درجة الثقة: {confidence:.1f}%** ✅")
                    else:
                        st.warning(f"**درجة الثقة: {confidence:.1f}%** ⚠️")
                
                st.info(f"**دقة النموذج: {accuracy*100:.2f}%**")
                
                # توصيات
                st.write("### 💡 التوصيات:")
                if prediction == 1 and confidence >= confidence_threshold:
                    st.success("""
                    **إشارة شراء قوية:**
                    - اتجاه صاعد مع ثقة عالية
                    - فرصة جيدة للدخول في صفقة
                    - ضع وقف الخسارة عند 2-3%
                    """)
                elif prediction == 0 and confidence >= confidence_threshold:
                    st.error("""
                    **إشارة بيع قوية:**
                    - اتجاه هابط مع ثقة عالية
                    - تجنب الشراء حالياً
                    - فرصة للدخول في صفقات بيع
                    """)
                else:
                    st.warning("""
                    **إشارة محايدة:**
                    - الثقة غير كافية
                    - الانتظار أفضل خيار
                    - ابحث عن تأكيدات إضافية
                    """)
            
            # تحليل الصورة
            if uploaded_file is not None:
                st.write("### 📷 تحليل الصورة المتقدم:")
                image_pred, image_score = analyze_image_advanced(uploaded_file)
                
                if image_pred == 1:
                    st.success(f"**نتيجة الصورة: 📈 إيجابية (درجة: {image_score}/4)**")
                elif image_pred == 0:
                    st.error(f"**نتيجة الصورة: 📉 سلبية (درجة: {image_score}/4)**")
            
            # عرض البيانات
            with st.expander("📋 عرض البيانات الكاملة"):
                st.dataframe(data.tail(10))
                
                # رسم بياني
                st.write("**آخر 100 يوم تداول:**")
                st.line_chart(data['Close'].tail(100))

        except Exception as e:
            st.error(f"❌ حدث خطأ غير متوقع: {str(e)}")
            st.stop()

st.markdown("---")
st.write("### 📝 ملاحظات مهمة:")
st.info("""
- استخدم بيانات تاريخية طويلة (سنتين على الأقل)
- درجة الثقة مهمة لقوة الإشارة
- لا تعتمد على إشارة واحدة فقط
- هذه أداة تعليمية واستشارية
- استشر متخصصاً قبل التداول الفعلي
""")

# ملف requirements.txt المطلوب
st.sidebar.markdown("---")
st.sidebar.info("""
**المكتبات المطلوبة:**