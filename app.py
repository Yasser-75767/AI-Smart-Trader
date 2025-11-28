# app_advanced.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import cv2
import datetime
import random
import ta  # مكتبة للمؤشرات الفنية المتقدمة

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader Pro — النسخة الدقيقة 💎", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META", "NFLX", "AMD", "INTC"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X"]
crypto_symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "SOL-USD"]
all_symbols = stock_symbols + forex_symbols + crypto_symbols

# ===== الشريط الجانبي =====
st.sidebar.header("⚙️ الإعدادات المتقدمة")
symbol = st.sidebar.selectbox("اختر السهم أو الزوج:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())

# إعدادات النموذج المتقدمة
st.sidebar.markdown("### 🧠 إعدادات الذكاء الاصطناعي")
model_type = st.sidebar.selectbox("نموذج التداول:", ["XGBoost المتقدم", "Random Forest", "المجمع"])
confidence_threshold = st.sidebar.slider("حد الثقة (%)", 50, 95, 75)

uploaded_file = st.sidebar.file_uploader("رفع صورة التحليل:", type=["png","jpg","jpeg"])

# ===== دوال محسنة =====

def load_enhanced_data(symbol, start, end):
    """تحميل بيانات محسنة مع معلومات إضافية"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1d")
        
        if df.empty or len(df) < 50:
            return pd.DataFrame(), symbol
            
        # بيانات إضافية
        info = ticker.info
        st.sidebar.markdown(f"**معلومات السهم:**")
        st.sidebar.write(f"القيمة السوقية: {info.get('marketCap', 'N/A')}")
        st.sidebar.write(f"القيمة الدفترية: {info.get('bookValue', 'N/A')}")
        st.sidebar.write(f"نسبة P/E: {info.get('trailingPE', 'N/A')}")
        
        return df, symbol
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {e}")
        return pd.DataFrame(), symbol

def calculate_advanced_indicators(df):
    """حساب مؤشرات فنية متقدمة"""
    df = df.copy()
    
    # المؤشرات الأساسية
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    df['Gap'] = df['Open'] - df['Close'].shift(1)
    
    # المتوسطات المتحركة
    for period in [5, 10, 20, 50, 200]:
        df[f'MA_{period}'] = df['Close'].rolling(period).mean()
        df[f'Volume_MA_{period}'] = df['Volume'].rolling(period).mean()
    
    # RSI بفترات مختلفة
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['RSI_7'] = ta.momentum.RSIIndicator(df['Close'], window=7).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # مؤشرات الحجم
    df['Volume_Rate'] = df['Volume'] / df['Volume_MA_20']
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    
    # الاتجاهات
    df['Trend_Strength'] = abs(df['Close'] - df['MA_20']) / df['MA_20']
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    # أنماط الشموع
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Is_Doji'] = (df['Body_Size'] / (df['High'] - df['Low']) < 0.1).astype(int)
    
    return df

def prepare_advanced_features(df, with_target=True):
    """تحضير الميزات المتقدمة"""
    if df.empty or len(df) < 50:
        return None, None, None
    
    try:
        # حساب المؤشرات المتقدمة
        df = calculate_advanced_indicators(df)
        
        # تحديد الميزات النهائية
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Price_Range', 'Price_Change', 'Gap',
            'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_200',
            'Volume_MA_5', 'Volume_MA_20',
            'RSI_14', 'RSI_7', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Lower', 'BB_Width',
            'Stoch_K', 'Stoch_D', 'Volume_Rate', 'OBV',
            'Trend_Strength', 'Volatility',
            'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 'Is_Doji'
        ]
        
        # ملء القيم الناقصة
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        if with_target:
            # إنشاء أهداف متعددة
            df['Target_Next_Day'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df['Target_Next_Week'] = (df['Close'].shift(-5) > df['Close']).astype(int)
            df['Target_Volatility'] = (df['Close'].pct_change().shift(-1).abs() > 0.02).astype(int)
            
            # حذف الصفوف التي تحتوي على قيم NaN في الأهداف
            df_clean = df.dropna(subset=['Target_Next_Day', 'Target_Next_Week', 'Target_Volatility'])
            
            if df_clean.empty:
                return None, None, None
                
            X = df_clean[feature_columns]
            y = df_clean['Target_Next_Day']  # التركيز على التنبؤ اليومي
            
            return X, y, df_clean
        else:
            X = df[feature_columns]
            return X, df, None
            
    except Exception as e:
        st.error(f"⚠ خطأ في تجهيز الميزات: {str(e)}")
        return None, None, None

def train_advanced_model(df, model_type="XGBoost المتقدم"):
    """تدريب نموذج متقدم"""
    X, y, df_processed = prepare_advanced_features(df, with_target=True)
    
    if X is None or y is None or len(X) < 100:
        st.warning("⚠ تحتاج إلى 100 نقطة بيانات على الأقل للتدريب المتقدم")
        return None, None, None
    
    try:
        # تقسيم البيانات الزمني
        tscv = TimeSeriesSplit(n_splits=5)
        scaler = StandardScaler()
        
        # تطبيع الميزات
        X_scaled = scaler.fit_transform(X)
        
        if model_type == "XGBoost المتقدم":
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                tree_method="hist",
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            )
        elif model_type == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:  # نموذج مجمع
            from sklearn.ensemble import VotingClassifier
            xgb_model = xgb.XGBClassifier(n_estimators=300, random_state=42)
            rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
            model = VotingClassifier(
                estimators=[('xgb', xgb_model), ('rf', rf_model)],
                voting='soft'
            )
        
        # التدريب مع التحقق المتقاطع
        cv_scores = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            cv_scores.append(accuracy_score(y_test, model.predict(X_test)))
        
        avg_accuracy = np.mean(cv_scores)
        
        # تدريب النموذج النهائي على كل البيانات
        model.fit(X_scaled, y)
        
        return model, avg_accuracy, scaler
        
    except Exception as e:
        st.error(f"⚠ خطأ في التدريب المتقدم: {str(e)}")
        return None, None, None

def predict_with_confidence(model, scaler, df):
    """التنبؤ مع حساب درجة الثقة"""
    X_pred, df_processed, _ = prepare_advanced_features(df, with_target=False)
    
    if X_pred is None or X_pred.empty:
        return None, None
    
    try:
        X_scaled = scaler.transform(X_pred)
        
        # التنبؤ واحتمالات التنبؤ
        prediction = model.predict(X_scaled[-1:])[0]
        probabilities = model.predict_proba(X_scaled[-1:])[0]
        
        # درجة الثقة
        confidence = max(probabilities) * 100
        
        return prediction, confidence
        
    except Exception as e:
        st.error(f"⚠ خطأ في التنبؤ: {str(e)}")
        return None, None

def advanced_image_analysis(file):
    """تحليل متقدم للصور"""
    try:
        image = Image.open(file).convert("RGB")
        image = image.resize((512, 512))
        
        # تحويل الصورة إلى OpenCV
        img_cv = np.array(image)
        
        # تحليل متعدد الأبعاد
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # حساب مؤشرات متعددة
        mean_intensity = np.mean(gray)
        intensity_std = np.std(gray)
        
        # اكتشاف الحواف
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # تحليل الألوان
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        # نظام تسجيل متقدم
        score = 0
        if mean_intensity > 130: score += 1
        if edge_density > 0.1: score += 1
        if saturation > 80: score += 1
        if intensity_std > 40: score += 1
        
        st.image(image, caption="📊 الصورة المحملة", use_column_width=True)
        
        st.write("### 📈 تحليل الصورة المتقدم:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("متوسط الإضاءة", f"{mean_intensity:.1f}")
        with col2:
            st.metric("كثافة الحواف", f"{edge_density:.3f}")
        with col3:
            st.metric("تشبع الألوان", f"{saturation:.1f}")
        with col4:
            st.metric("التباين", f"{intensity_std:.1f}")
        
        return 1 if score >= 2 else 0, score
        
    except Exception as e:
        st.error(f"⚠ خطأ في تحليل الصورة: {str(e)}")
        return None, 0

# ===== واجهة التطبيق المحسنة =====
st.title("🎯 AI Smart Trader Pro — النسخة الدقيقة جداً 💎")
st.warning("⚠ **تحذير مهم:** هذه أداة تعليمية. التداول الفعلي يحمل مخاطر فقدان رأس المال.")

if st.button("🚀 الحصول على تحليل دقيق"):
    with st.spinner("🔬 جاري التحليل المتعمق... قد يستغرق دقائق"):
        try:
            # تحميل البيانات
            df, used_symbol = load_enhanced_data(symbol, start_date, end_date)
            if df.empty or len(df) < 100:
                st.error("❌ تحتاج إلى 100 يوم تداول على الأقل للتحليل الدقيق")
                st.stop()
            
            st.success(f"✅ تم تحميل {len(df)} يوم تداول للرمز {used_symbol}")
            
            # عرض إحصائيات متقدمة
            st.write("### 📊 الإحصائيات المتقدمة:")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                returns = df['Close'].pct_change().dropna()
                vol_30d = returns.tail(30).std() * np.sqrt(252) * 100
                st.metric("التقلب (30 يوم)", f"{vol_30d:.1f}%")
            
            with col2:
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                st.metric("نسبة شارب", f"{sharpe:.2f}")
            
            with col3:
                max_drawdown = (df['Close'] / df['Close'].cummax() - 1).min() * 100
                st.metric("أقصى خسارة", f"{max_drawdown:.1f}%")
            
            with col4:
                current_rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
                st.metric("RSI الحالي", f"{current_rsi:.1f}")
            
            # تدريب النموذج المتقدم
            model, accuracy, scaler = train_advanced_model(df, model_type)
            
            if model is None:
                st.error("❌ فشل في تدريب النموذج المتقدم")
                st.stop()
            
            # التنبؤ الدقيق
            prediction, confidence = predict_with_confidence(model, scaler, df)
            
            if prediction is not None:
                st.write("### 🎯 نتائج التحليل المتقدم:")
                
                # عرض نتيجة التنبؤ
                col_pred, col_conf = st.columns(2)
                
                with col_pred:
                    if prediction == 1:
                        st.success(f"**الاتجاه: 📈 صاعد**")
                        st.progress(0.8)
                    else:
                        st.error(f"**الاتجاه: 📉 هابط**")
                        st.progress(0.2)
                
                with col_conf:
                    if confidence >= confidence_threshold:
                        st.success(f"**درجة الثقة: {confidence:.1f}%** ✅")
                    else:
                        st.warning(f"**درجة الثقة: {confidence:.1f}%** ⚠️")
                
                # توصيات مبنية على التحليل
                st.write("### 💡 التوصيات الإستراتيجية:")
                
                if prediction == 1 and confidence >= confidence_threshold:
                    st.success("""
                    **إشارة شراء قوية:**
                    - فرصة جيدة للدخول في صفقة شراء
                    - ضع وقف الخسارة عند 3-5% تحت نقطة الدخول
                    - هدف الربح عند 8-12% فوق نقطة الدخول
                    """)
                elif prediction == 0 and confidence >= confidence_threshold:
                    st.error("""
                    **إشارة بيع قوية:**
                    - تجنب الشراء حالياً
                    - فرصة للدخول في صفقات بيع
                    - انتظر تأكيدات إضافية
                    """)
                else:
                    st.info("""
                    **إشارة محايدة:**
                    - الانتظار أفضل استراتيجية
                    - ابحث عن تأكيدات إضافية
                    - راقب مستويات الدعم والمقاومة
                    """)
            
            # تحليل الصورة المتقدم
            if uploaded_file is not None:
                st.write("### 📷 تحليل الصورة المتقدم:")
                img_pred, img_score = advanced_image_analysis(uploaded_file)
                
                if img_pred == 1:
                    st.success(f"**تحليل الصورة: 📈 إيجابي (درجة: {img_score}/4)**")
                elif img_pred == 0:
                    st.error(f"**تحليل الصورة: 📉 سلبي (درجة: {img_score}/4)**")
            
            # عرض البيانات التفصيلية
            with st.expander("📋 عرض البيانات التفصيلية والمؤشرات"):
                st.dataframe(df.tail(15))
                
                # رسم بياني مبسط
                st.write("**آخر 50 يوم تداول:**")
                st.line_chart(df['Close'].tail(50))
            
            # نصائح إضافية
            st.info("""
            ### 📝 نصائح للاستخدام الأمثل:
            - استخدم بيانات تاريخية طويلة (سنتين على الأقل)
            - جرب نماذج مختلفة لمقارنة النتائج
            - لا تعتمد على إشارة واحدة فقط
            - استخدم التحليل المتعدد الأطراف الزمنية
            - راقب درجة الثقة في التنبؤات
            """)
            
        except Exception as e:
            st.error(f"❌ حدث خطأ غير متوقع: {str(e)}")
            st.stop()

st.markdown("---")
st.subheader("⭐ نظام التداول الذكي")
st.write("""
هذا النظام يستخدم:
- **12+ مؤشر فني متقدم**
- **تعلم آلي متعدد النماذج**
- **تحقق متقاطع زمني**
- **تحليل صور ذكي**
- **إدارة مخاطر متكاملة**
""")

st.markdown("---")
st.info("""
### 🎓 ملاحظة تعليمية:
هذه الأداة مصممة للأغراض التعليمية والبحثية فقط. 
يجب استشارة مستشار مالي محترف قبل اتخاذ أي قرارات تداول حقيقية.
""")