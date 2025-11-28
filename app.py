# app_fixed.py
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
    page_title="AI Smart Trader — النسخة الهاتفية 💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== الرموز =====
stock_symbols = ["AAPL","MSFT","GOOGL","NVDA","AMZN","TSLA","META","NFLX"]
forex_symbols = ["EURUSD=X","USDJPY=X","GBPUSD=X","USDCHF=X","AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols

# ===== الشريط الجانبي =====
st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر السهم أو الزوج:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2020,1,1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
confidence_threshold = st.sidebar.slider("حد الثقة (%)",50,95,80)
uploaded_file = st.sidebar.file_uploader("رفع صورة الشموع/المنحنيات", type=["png","jpg","jpeg"])

# ===== الحاويات الثابتة =====
stats_container = st.container()
indicators_container = st.container()
prediction_container = st.container()
image_container = st.container()
data_container = st.expander("📋 عرض البيانات الكاملة")

# ===== الدوال =====
def load_data(symbol,start,end):
    try:
        df = yf.download(symbol,start=start,end=end,progress=False)
        if df.empty or len(df)<100:
            st.error("❌ البيانات غير كافية. تحتاج 100 يوم على الأقل")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"❌ خطأ في التحميل: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    df = df.copy()
    for period in [5,10,20,50]:
        df[f'MA_{period}'] = df['Close'].rolling(period,min_periods=1).mean()
    try:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'],window=14).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
    except:
        df['RSI']=0
        df['MACD']=0
        df['MACD_Signal']=0
        df['BB_Upper']=0
        df['BB_Lower']=0
    df['Volatility'] = df['Close'].pct_change().rolling(20,min_periods=1).std()
    df['Price_Range'] = df['High']-df['Low']
    df['Price_Change'] = df['Close']-df['Open']
    df['Gap'] = df['Open']-df['Close'].shift(1)
    df['Volume_MA'] = df['Volume'].rolling(20,min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume']/df['Volume_MA'].replace(0,1)
    return df.fillna(0)

def prepare_features(df,with_target=True):
    if df.empty or len(df)<50:
        return None,None,None
    df = calculate_indicators(df)
    features = ['Open','High','Low','Close','Volume','MA_5','MA_20','MA_50',
                'RSI','MACD','MACD_Signal','BB_Upper','BB_Lower','Volume_Ratio',
                'Volatility','Price_Range','Price_Change','Gap']
    for feat in features:
        if feat not in df.columns:
            df[feat]=0
    if with_target:
        df['Target'] = (df['Close'].shift(-1)>df['Close']).astype(int)
        clean = df.iloc[:-1].copy()
        if clean.empty:
            return None,None,None
        X = clean[features]
        y = clean['Target']
        return X,y,clean
    else:
        X = df[features]
        return X,df,None

def train_model(df):
    X,y,processed = prepare_features(df,with_target=True)
    if X is None or len(X)<100:
        st.warning("⚠ تحتاج 100 نقطة بيانات على الأقل للتدريب")
        return None,None,None
    tscv = TimeSeriesSplit(n_splits=3)
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
    scores=[]
    for train_idx,test_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx],y.iloc[train_idx])
        scores.append(accuracy_score(y.iloc[test_idx],model.predict(X_scaled[test_idx])))
    avg_acc = np.mean(scores)
    model.fit(X_scaled,y)
    return model,avg_acc,scaler

def predict(model,scaler,df):
    X,processed,_ = prepare_features(df,with_target=False)
    if X is None:
        return None,None
    X_scaled = scaler.transform(X)
    last_row = X_scaled[-1:].reshape(1,-1)
    pred = model.predict(last_row)[0]
    conf = float(max(model.predict_proba(last_row)[0]))*100
    return pred,conf

def analyze_image(file):
    try:
        image = Image.open(file).convert("RGB").resize((300,300))
        gray = image.convert('L')
        stat = ImageStat.Stat(gray)
        mean_brightness=float(stat.mean[0])
        edges = image.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges.convert('L'))
        edge_intensity=float(edge_stat.mean[0])
        contrast = image.filter(ImageFilter.CONTOUR)
        contrast_stat = ImageStat.Stat(contrast.convert('L'))
        contrast_level=float(contrast_stat.mean[0])
        score=0
        if mean_brightness>130: score+=1
        if edge_intensity>30: score+=1
        if contrast_level>50: score+=1
        return 1 if score>=2 else 0,score
    except:
        return None,0

# ===== واجهة التطبيق =====
st.title("🎯 AI Smart Trader — نسخة الهاتف 💎")
st.warning("⚠ أداة تعليمية. التداول الفعلي يحمل مخاطر.")

if st.button("🚀 بدء التحليل"):
    with st.spinner("🔬 جاري التحليل..."):
        df = load_data(symbol,start_date,end_date)
        if df.empty:
            st.stop()
        
        # إحصائيات أساسية
        with stats_container:
            st.subheader("📊 الإحصائيات الأساسية")
            st.metric("متوسط الإغلاق",f"{float(df['Close'].mean()):.2f}")
            st.metric("أعلى سعر",f"{float(df['High'].max()):.2f}")
            st.metric("أقل سعر",f"{float(df['Low'].min()):.2f}")
        
        # مؤشرات وتنبؤ
        model,acc,scaler = train_model(df)
        if model is None:
            st.error("❌ تعذر تدريب النموذج")
            st.stop()
        
        pred,conf = predict(model,scaler,df)
        with prediction_container:
            st.subheader("🎯 التنبؤ والاتجاه")
            if pred==1:
                st.success("📈 الاتجاه: صاعد")
            else:
                st.error("📉 الاتجاه: هابط")
            if conf>=confidence_threshold:
                st.success(f"درجة الثقة: {conf:.1f}% ✅")
            else:
                st.warning(f"درجة الثقة: {conf:.1f}% ⚠️")
            st.info(f"دقة النموذج: {acc*100:.2f}%")
        
        # تحليل الصورة
        if uploaded_file is not None:
            with image_container:
                img_pred,img_score=analyze_image(uploaded_file)
                st.subheader("📷 تحليل الصورة")
                if img_pred==1:
                    st.success(f"📈 إيجابية (درجة: {img_score}/3)")
                elif img_pred==0:
                    st.error(f"📉 سلبية (درجة: {img_score}/3)")
                else:
                    st.info("تعذر تحليل الصورة")
        
        # عرض البيانات
        with data_container:
            st.dataframe(df.tail(10))
            st.line_chart(df['Close'].tail(100))