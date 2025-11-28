# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from PIL import Image, ImageStat, ImageFilter
import datetime
import ta

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader — الهاتف 💎", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META", "NFLX"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر السهم أو الزوج:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2020,1,1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
confidence_threshold = st.sidebar.slider("حد الثقة (%)", 50, 95, 80)
uploaded_file = st.sidebar.file_uploader("رفع صورة الشموع/المنحنيات", type=["png","jpg","jpeg"])

# ===== دوال التحليل =====
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty or len(df) < 100:
        st.error("❌ البيانات غير كافية (أقل من 100 يوم)")
        return pd.DataFrame()
    return df

def compute_indicators(df):
    df = df.copy()
    # المتوسطات المتحركة
    for p in [5, 20, 50]:
        df[f"MA_{p}"] = df['Close'].rolling(p, min_periods=1).mean()
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    # الحجم والتقلب
    df['Volume_MA'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(20, min_periods=1).std()
    # السعر
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    df['Gap'] = df['Open'] - df['Close'].shift(1)
    return df.fillna(0)

def prepare_features(df, with_target=True):
    df = compute_indicators(df)
    features = ['Open','High','Low','Close','Volume','MA_5','MA_20','MA_50',
                'RSI','MACD','MACD_Signal','BB_Upper','BB_Lower','Volume_MA',
                'Volatility','Price_Range','Price_Change','Gap']
    if with_target:
        df["Target"] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.iloc[:-1]
        X = df[features]
        y = df["Target"]
        return X, y
    else:
        X = df[features]
        return X

def train_model(df):
    X, y = prepare_features(df)
    if len(X) < 100:
        st.warning("⚠ بيانات أقل من 100 نقطة لا تكفي للتدريب")
        return None, None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=3)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                              learning_rate=0.1, use_label_encoder=False,
                              eval_metric="logloss", tree_method="hist",
                              random_state=42)
    accuracies = []
    for train_idx, test_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X_scaled[test_idx])
        accuracies.append(accuracy_score(y.iloc[test_idx], y_pred))
    model.fit(X_scaled, y)
    return model, scaler, np.mean(accuracies)

def predict(model, scaler, df):
    X = prepare_features(df, with_target=False)
    X_scaled = scaler.transform(X)
    last = X_scaled[-1].reshape(1,-1)
    pred = model.predict(last)[0]
    confidence = max(model.predict_proba(last)[0]) * 100
    return pred, confidence

def analyze_image(file):
    image = Image.open(file).convert("RGB").resize((400,400))
    st.image(image, caption="📊 الصورة", width=300)
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    mean_brightness = float(stat.mean[0])
    std_brightness = float(stat.stddev[0])
    edges = image.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges.convert("L"))
    edge_intensity = float(edge_stat.mean[0])
    contrast = image.filter(ImageFilter.CONTOUR)
    contrast_stat = ImageStat.Stat(contrast.convert("L"))
    contrast_level = float(contrast_stat.mean[0])
    score = sum([mean_brightness>130, edge_intensity>30, std_brightness>40, contrast_level>50])
    return 1 if score>=2 else 0, score, mean_brightness, edge_intensity, std_brightness, contrast_level

# ===== واجهة التطبيق =====
st.title("🎯 AI Smart Trader — الهاتف 💎")
st.warning("⚠ أداة تعليمية، التداول الفعلي يحمل مخاطر")

if st.button("🚀 تحليل شامل"):
    df = load_data(symbol, start_date, end_date)
    if df.empty:
        st.stop()
    st.success(f"✅ تم تحميل {len(df)} يوم تداول لـ {symbol}")
    
    st.write("### 📊 إحصائيات أساسية")
    col1, col2, col3 = st.columns(3)
    col1.metric("متوسط الإغلاق", f"{df['Close'].mean():.2f}")
    col2.metric("أعلى سعر", f"{df['High'].max():.2f}")
    col3.metric("أقل سعر", f"{df['Low'].min():.2f}")
    
    st.write("### 📈 المؤشرات الفنية")
    df = compute_indicators(df)
    col4, col5, col6 = st.columns(3)
    col4.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    col5.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
    col6.metric("BB Upper", f"{df['BB_Upper'].iloc[-1]:.2f}")
    
    model, scaler, acc = train_model(df)
    if model:
        pred, confidence = predict(model, scaler, df)
        st.write("### 🎯 التنبؤ النهائي")
        col7, col8 = st.columns(2)
        col7.metric("الاتجاه", "📈 صاعد" if pred==1 else "📉 هابط")
        col8.metric("درجة الثقة", f"{confidence:.1f}%")
        st.info(f"دقة النموذج: {acc*100:.2f}%")
    
    if uploaded_file:
        img_pred, score, mean_brightness, edge_intensity, std_brightness, contrast_level = analyze_image(uploaded_file)
        st.write("### 📷 تحليل الصورة")
        st.metric("نتيجة الصورة", "📈 إيجابية" if img_pred==1 else "📉 سلبية")
        st.metric("درجة التحليل", f"{score}/4")
        st.metric("متوسط الإضاءة", f"{mean_brightness:.1f}")
        st.metric("شدة الحواف", f"{edge_intensity:.1f}")
        st.metric("التباين", f"{std_brightness:.1f}")
        st.metric("التفاصيل", f"{contrast_level:.1f}")
    
    st.line_chart(df['Close'].tail(100))