# app_final_fixed.py — AI Smart Trader النسخة الدقيقة جداً 💎
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
st.set_page_config(
    page_title="AI Smart Trader — النسخة الدقيقة 💎", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ===== دوال التحليل الفني =====
def calculate_advanced_indicators(data):
    data = data.copy()
    data['Close'] = data['Close'].fillna(method='ffill').astype(float)
    data['High'] = data['High'].fillna(method='ffill').astype(float)
    data['Low'] = data['Low'].fillna(method='ffill').astype(float)
    data['Open'] = data['Open'].fillna(method='ffill').astype(float)
    data['Volume'] = data['Volume'].fillna(0).astype(float)
    
    try:
        # المتوسطات المتحركة
        for period in [5, 10, 20, 50]:
            data[f'MA_{period}'] = data['Close'].rolling(period, min_periods=1).mean()
        
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
        data['Volume_MA'] = data['Volume'].rolling(20, min_periods=1).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA'].replace(0, 1)
        
        # التقلب
        data['Volatility'] = data['Close'].pct_change().rolling(20, min_periods=1).std()
        
        # أنماط السعر
        data['Price_Range'] = data['High'] - data['Low']
        data['Price_Change'] = data['Close'] - data['Open']
        data['Gap'] = data['Open'] - data['Close'].shift(1)
        
    except Exception as e:
        st.error(f"⚠ خطأ في حساب المؤشرات: {e}")
    
    return data.fillna(0)

def prepare_features(data, with_target=True):
    data = calculate_advanced_indicators(data)
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'RSI', 'MACD', 'MACD_Signal',
                'BB_Upper', 'BB_Lower',
                'Volume_Ratio', 'Volatility',
                'Price_Range', 'Price_Change', 'Gap']
    
    if with_target:
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data_clean = data.iloc[:-1]
        X = data_clean[features]
        y = data_clean['Target']
        return X, y, data_clean
    else:
        X = data[features]
        return X, data, None

def train_model(data):
    X, y, _ = prepare_features(data, with_target=True)
    if X is None or len(X) < 100:
        st.warning("⚠ تحتاج بيانات أكثر للتدريب")
        return None, None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        tree_method="hist",
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, test_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X_scaled[test_idx])
        scores.append(accuracy_score(y.iloc[test_idx], y_pred))
    model.fit(X_scaled, y)
    return model, np.mean(scores), scaler

def predict_last(model, scaler, data):
    X_pred, _, _ = prepare_features(data, with_target=False)
    if X_pred is None or X_pred.empty:
        return None, None
    X_scaled = scaler.transform(X_pred)
    last_row = X_scaled[-1:].reshape(1, -1)
    pred = model.predict(last_row)[0]
    conf = float(max(model.predict_proba(last_row)[0]))*100
    return pred, conf

def analyze_image(file):
    try:
        image = Image.open(file).convert("RGB").resize((400,400))
        st.image(image, caption="📊 الصورة المحملة", use_column_width=False)
        gray = image.convert('L')
        stat = ImageStat.Stat(gray)
        mean = float(stat.mean[0])
        std = float(stat.stddev[0])
        edges = image.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges.convert('L'))
        edge_mean = float(edge_stat.mean[0])
        contrast = image.filter(ImageFilter.CONTOUR)
        contrast_stat = ImageStat.Stat(contrast.convert('L'))
        contrast_mean = float(contrast_stat.mean[0])
        score = sum([mean>130, std>40, edge_mean>30, contrast_mean>50])
        st.write("**📈 تحليل الصورة المتقدم:**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("متوسط الإضاءة", f"{mean:.1f}")
        col2.metric("التباين", f"{std:.1f}")
        col3.metric("شدة الحواف", f"{edge_mean:.1f}")
        col4.metric("التفاصيل", f"{contrast_mean:.1f}")
        return 1 if score>=2 else 0, score
    except Exception as e:
        st.error(f"⚠ خطأ في تحليل الصورة: {e}")
        return None, 0

# ===== واجهة التطبيق =====
st.title("🎯 AI Smart Trader — النسخة الدقيقة 💎")
st.warning("⚠ أداة تعليمية فقط. التداول الحقيقي يحمل مخاطر مالية.")

if st.button("🚀 بدء التحليل الدقيق"):
    with st.spinner("🔬 جاري التحليل المتقدم..."):
        data, used_symbol = None, None
        try:
            data, used_symbol = yf.download(symbol, start=start_date, end=end_date, progress=False), symbol
            if data.empty:
                st.error("❌ البيانات غير كافية")
                st.stop()
            
            st.success(f"✅ تم تحميل {len(data)} يوم تداول لـ {used_symbol}")
            
            # إحصائيات أساسية
            st.write("### 📊 الإحصائيات الأساسية:")
            col1, col2, col3 = st.columns(3)
            col1.metric("متوسط الإغلاق", f"{float(data['Close'].mean()):.2f}")
            col2.metric("أعلى سعر", f"{float(data['High'].max()):.2f}")
            col3.metric("أقل سعر", f"{float(data['Low'].min()):.2f}")
            
            # تدريب النموذج
            model, accuracy, scaler = train_model(data)
            if model is None:
                st.error("❌ تعذر تدريب النموذج")
                st.stop()
            
            # التنبؤ
            pred, conf = predict_last(model, scaler, data)
            if pred is not None:
                st.write("### 🎯 نتائج التحليل:")
                col1, col2 = st.columns(2)
                col1.metric("الاتجاه", "📈 صاعد" if pred==1 else "📉 هابط")
                col2.metric("درجة الثقة (%)", f"{conf:.1f}")
                st.info(f"دقة النموذج: {accuracy*100:.2f}%")
            
            # تحليل الصورة
            if uploaded_file:
                st.write("### 📷 تحليل الصورة:")
                image_pred, image_score = analyze_image(uploaded_file)
                if image_pred==1:
                    st.success(f"📈 نتيجة الصورة: إيجابية (درجة: {image_score}/4)")
                else:
                    st.error(f"📉 نتيجة الصورة: سلبية (درجة: {image_score}/4)")
            
            # عرض البيانات
            with st.expander("📋 عرض البيانات الأخيرة"):
                st.dataframe(data.tail(10))
                st.line_chart(data['Close'].tail(100))
                
        except Exception as e:
            st.error(f"❌ حدث خطأ غير متوقع: {e}")

st.markdown("---")
st.sidebar.info("المكتبات المطلوبة: streamlit, yfinance, pandas, numpy, xgboost, scikit-learn, pillow, ta")