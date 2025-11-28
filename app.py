# app_final_mobile.py
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
import ta
import random

# ===== إعداد الصفحة =====
st.set_page_config(
    page_title="AI Smart Trader — الهاتف 💎",
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
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2020,1,1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
confidence_threshold = st.sidebar.slider("حد الثقة (%)", 50, 95, 80)
uploaded_file = st.sidebar.file_uploader("رفع صورة الشموع/المنحنيات للتحليل", type=["png","jpg","jpeg"])

# ===== الدوال =====
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        if data.empty or len(data) < 100:
            st.error("❌ البيانات غير كافية. تحتاج 100 يوم على الأقل")
            return pd.DataFrame(), symbol
        return data, symbol
    except Exception as e:
        st.error(f"❌ خطأ في التحميل: {e}")
        return pd.DataFrame(), symbol

def calculate_indicators(data):
    data = data.copy()
    # المتوسطات المتحركة
    for period in [5, 20, 50]:
        data[f"MA_{period}"] = data['Close'].rolling(period, min_periods=1).mean()
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
    data['Volume_Ratio'] = data['Volume']/data['Volume_MA'].replace(0,1)
    # التقلب
    data['Volatility'] = data['Close'].pct_change().rolling(20, min_periods=1).std()
    # أنماط السعر
    data['Price_Range'] = data['High']-data['Low']
    data['Price_Change'] = data['Close']-data['Open']
    data['Gap'] = data['Open']-data['Close'].shift(1)
    return data.fillna(0)

def prepare_features(data, with_target=True):
    if data.empty or len(data) < 50:
        return None, None, None
    data = calculate_indicators(data)
    features = ['Open','High','Low','Close','Volume','MA_5','MA_20','MA_50','RSI','MACD','MACD_Signal',
                'BB_Upper','BB_Lower','Volume_Ratio','Volatility','Price_Range','Price_Change','Gap']
    for feat in features:
        if feat not in data.columns:
            data[feat]=0
    if with_target:
        data["Target"]=(data['Close'].shift(-1)>data['Close']).astype(int)
        clean_data=data.iloc[:-1].copy()
        X=clean_data[features]
        y=clean_data['Target']
        return X,y,clean_data
    else:
        X=data[features]
        return X,data,None

def train_model(data):
    X,y,_=prepare_features(data, with_target=True)
    if X is None or len(X)<100:
        st.warning("⚠ تحتاج 100 نقطة بيانات على الأقل للتدريب")
        return None,None,None
    tscv=TimeSeriesSplit(n_splits=3)
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    model=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,
                            tree_method="hist",use_label_encoder=False,eval_metric="logloss",random_state=42)
    scores=[]
    for train_idx,test_idx in tscv.split(X_scaled):
        X_train,X_test=X_scaled[train_idx],X_scaled[test_idx]
        y_train,y_test=y.iloc[train_idx],y.iloc[test_idx]
        model.fit(X_train,y_train)
        scores.append(accuracy_score(y_test,model.predict(X_test)))
    avg_acc=np.mean(scores)
    model.fit(X_scaled,y)
    return model,avg_acc,scaler

def predict_last(model,scaler,data):
    X_pred,_,_=prepare_features(data, with_target=False)
    if X_pred is None or X_pred.empty:
        return None,None
    X_scaled=scaler.transform(X_pred)
    last_row=X_scaled[-1:].reshape(1,-1)
    pred=model.predict(last_row)[0]
    conf=max(model.predict_proba(last_row)[0])*100
    return pred,conf

def analyze_image(file):
    try:
        image=Image.open(file).convert("RGB").resize((400,400))
        st.image(image,width=300)
        gray=ImageStat.Stat(image.convert('L'))
        mean_brightness=float(gray.mean[0])
        score=1 if mean_brightness>130 else 0
        st.metric("متوسط الإضاءة",f"{mean_brightness:.1f}")
        return score
    except:
        return None

# ===== واجهة المستخدم =====
st.title("🎯 AI Smart Trader — الهاتف 💎")
st.warning("⚠ أداة تعليمية فقط")

placeholder_main=st.empty()
placeholder_chart=st.empty()

if st.button("🚀 بدء التحليل"):
    with placeholder_main.container():
        data,used_symbol=load_data(symbol,start_date,end_date)
        if data.empty: st.stop()
        st.success(f"✅ تم تحميل {len(data)} يوم تداول لـ {used_symbol}")
        
        # إحصائيات
        col1,col2,col3=st.columns(3)
        col1.metric("متوسط الإغلاق",f"{float(data['Close'].mean()):.2f}")
        col2.metric("أعلى سعر",f"{float(data['High'].max()):.2f}")
        col3.metric("أقل سعر",f"{float(data['Low'].min()):.2f}")
        
        # تدريب النموذج والتنبؤ
        model,accuracy,scaler=train_model(data)
        if model is None: st.stop()
        pred,conf=predict_last(model,scaler,data)
        
        # عرض النتائج
        col4,col5=st.columns(2)
        if pred==1:
            col4.success("📈 الاتجاه: صاعد")
        else:
            col4.error("📉 الاتجاه: هابط")
        if conf>=confidence_threshold:
            col5.success(f"درجة الثقة: {conf:.1f}% ✅")
        else:
            col5.warning(f"درجة الثقة: {conf:.1f}% ⚠️")
        st.info(f"دقة النموذج: {accuracy*100:.2f}%")
        
        # تحليل الصورة
        if uploaded_file:
            image_result=analyze_image(uploaded_file)
            if image_result==1:
                st.success("📈 تحليل الصورة: إيجابي")
            elif image_result==0:
                st.error("📉 تحليل الصورة: سلبي")
            else:
                st.info("⚠ لم يتمكن التطبيق من تحليل الصورة")
        
        # رسم البيانات
        with placeholder_chart.container():
            st.line_chart(data['Close'].tail(100))
            st.dataframe(data.tail(10))