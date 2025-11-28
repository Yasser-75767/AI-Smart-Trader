# app.py — AI Smart Trader Pro (نسخة معدّلة مستقرة)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import ta
import datetime
import warnings
import random

warnings.filterwarnings("ignore")

# ================= إعداد الصفحة =================
st.set_page_config(page_title="AI Smart Trader Pro — النسخة المحسّنة", layout="wide")

# ================= القوائم =================
stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META", "JPM", "JNJ", "V", "WMT", "PG", "DIS", "NFLX", "ADBE"]
forex_pairs = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"]
crypto = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LTC-USD", "XRP-USD", "DOGE-USD", "SOL-USD", "AVAX-USD", "MATIC-USD"]
all_symbols = stocks + forex_pairs + crypto

# ================= الشريط الجانبي =================
st.sidebar.header("⚙️ الإعدادات المتقدمة")
symbol = st.sidebar.selectbox("اختر الأصل:", all_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())
model_type = st.sidebar.selectbox("نموذج التعلم الآلي:", ["XGBoost", "Random Forest", "Gradient Boosting", "Ensemble"])
lookback_days = st.sidebar.slider("أيام النظر للخلف:", 5, 40, 20)
test_size = st.sidebar.slider("حجم بيانات الاختبار (%):", 10, 40, 20)
confidence_threshold = st.sidebar.slider("حد الثقة لإشارة قوية (%):", 50, 95, 75)

if "run" not in st.session_state:
    st.session_state["run"] = False

# ================= دوال المساعدة =================
def fetch_market_sentiment():
    return {
        "bullish": random.uniform(0.4, 0.7),
        "bearish": random.uniform(0.2, 0.6),
        "neutral": random.uniform(0.1, 0.3),
    }

def load_enhanced_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    
    close = df["Close"].squeeze()
    open_ = df["Open"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    # مؤشرات فنية
    df["SMA_5"] = ta.trend.sma_indicator(close, window=5)
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    df["EMA_10"] = ta.trend.ema_indicator(close, window=10)
    df["EMA_50"] = ta.trend.ema_indicator(close, window=50)
    df["RSI_14"] = ta.momentum.rsi(close, window=14)
    df["MACD"] = ta.trend.macd(close)
    df["MACD_Signal"] = ta.trend.macd_signal(close)
    df["MACD_Hist"] = ta.trend.macd_diff(close)
    df["BB_Upper"] = ta.volatility.bollinger_hband(close)
    df["BB_Lower"] = ta.volatility.bollinger_lband(close)
    df["BB_Middle"] = ta.volatility.bollinger_mavg(close)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["Stoch_K"] = ta.momentum.stoch(high, low, close)
    df["Stoch_D"] = ta.momentum.stoch_signal(high, low, close)
    df["Williams_R"] = ta.momentum.williams_r(high, low, close)
    df["CCI"] = ta.trend.cci(high, low, close)
    df["ADX"] = ta.trend.adx(high, low, close)
    df["OBV"] = ta.volume.on_balance_volume(close, volume)
    df["ATR"] = ta.volatility.average_true_range(high, low, close)
    df["Returns"] = close.pct_change()
    df["Log_Returns"] = np.log(close / close.shift(1))
    df["Price_Range"] = (high - low) / close
    df["Gap"] = (open_ - close.shift(1)) / close.shift(1)
    df["Volatility_5D"] = df["Returns"].rolling(5).std()
    df["Volatility_21D"] = df["Returns"].rolling(21).std()
    df["Momentum_5D"] = close / close.shift(5) - 1
    df["Momentum_21D"] = close / close.shift(21) - 1
    df["Volume_SMA"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / df["Volume_SMA"].replace(0, np.nan)
    df["Day_of_Week"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter

    df = df.dropna()
    return df

def create_advanced_target(df, lookahead=1):
    future_return = df["Close"].shift(-lookahead) / df["Close"] - 1
    df["Binary_Target"] = (future_return > 0).astype(int)
    return df.dropna()

def prepare_features_for_ml(df, lookback_days):
    base_features = ["Close","SMA_5","SMA_20","SMA_50","EMA_10","EMA_50",
                     "RSI_14","MACD","MACD_Signal","MACD_Hist",
                     "BB_Upper","BB_Lower","BB_Width",
                     "Stoch_K","Stoch_D","Williams_R",
                     "CCI","ADX","OBV","ATR",
                     "Returns","Log_Returns","Price_Range","Gap",
                     "Volatility_5D","Volatility_21D",
                     "Momentum_5D","Momentum_21D",
                     "Volume_Ratio","Day_of_Week","Month","Quarter"]
    features = [f for f in base_features if f in df.columns]

    for f in features:
        for lag in range(1, min(lookback_days, 10)+1):
            df[f"{f}_lag_{lag}"] = df[f].shift(lag)

    df = df.dropna()
    X = df[[c for c in df.columns if c != "Binary_Target"]]
    y = df["Binary_Target"]
    return X, y, df

def train_advanced_model(X, y, model_type, test_size_ratio):
    split = int(len(X)*(1-test_size_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if model_type=="XGBoost":
        model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", use_label_encoder=False,
                                  random_state=42)
    elif model_type=="Random Forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    elif model_type=="Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    else:
        model = VotingClassifier([
            ("xgb", xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42))
        ], voting="soft")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

def generate_trading_signal(model, X_last_row, df_row, conf_threshold):
    try:
        proba = model.predict_proba(X_last_row)[0][1]
        pred = int(proba > 0.5)
        rsi = df_row.get("RSI_14",50)
        macd = df_row.get("MACD",0)
        vol_ratio = df_row.get("Volume_Ratio",1)

        if pred==1 and proba*100 >= conf_threshold:
            signal = "شراء قوي 🟢" if rsi<70 and macd>0 and vol_ratio>1 else "شراء 🟢"
        elif pred==0 and (1-proba)*100 >= conf_threshold:
            signal = "بيع قوي 🔴" if rsi>30 and macd<0 and vol_ratio>1 else "بيع 🔴"
        else:
            signal = "محايد ⚪"
        confidence = max(proba,1-proba)
        return signal, confidence
    except:
        return "خطأ في الإشارة",0.0

# ================= واجهة التطبيق =================
st.title("🎯 AI Smart Trader Pro — النسخة المحسّنة")
st.markdown("نظام متقدم للتحليل والتنبؤ المالي باستخدام الذكاء الاصطناعي.")
st.info("💡 هذا التطبيق تعليمي + عملي، يمكن استخدامه للتداول الفعلي (احذر المخاطر).")
run_button = st.button("🚀 بدء التحليل")
if run_button:
    st.session_state["run"]=True

if not st.session_state.get("run",False):
    st.warning("اضغط على زر **🚀 بدء التحليل** لبدء التحليل.")
else:
    with st.spinner("جاري تحميل البيانات..."):
        df = load_enhanced_data(symbol, start_date, end_date)
    if df.empty:
        st.error("❌ لا توجد بيانات كافية لهذا الأصل.")
    else:
        df = create_advanced_target(df)
        X, y, df_proc = prepare_features_for_ml(df, lookback_days)
        if len(X)<100:
            st.error("❌ البيانات بعد المعالجة غير كافية لتدريب نموذج.")
        else:
            with st.spinner("جاري تدريب النموذج..."):
                model, acc = train_advanced_model(X, y, model_type, test_size/100)
            st.success(f"✅ دقة النموذج: {acc*100:.2f}%")
            last_X = X.iloc[[-1]]
            last_row = df_proc.iloc[-1]
            signal, confidence = generate_trading_signal(model,last_X,last_row,confidence_threshold)
            
            col_sig, col_chart = st.columns([1,2])
            with col_sig:
                st.subheader("🎯 إشارة التداول الحالية")
                if "شراء قوي" in signal: st.success(f"{signal}\nثقة: {confidence*100:.1f}%")
                elif "بيع قوي" in signal: st.error(f"{signal}\nثقة: {confidence*100:.1f}%")
                elif "شراء" in signal: st.info(f"{signal}\nثقة: {confidence*100:.1f}%")
                elif "بيع" in signal: st.warning(f"{signal}\nثقة: {confidence*100:.1f}%")
                else: st.warning(f"{signal}\nثقة: {confidence*100:.1f}%")
            with col_chart:
                st.subheader("📈 السعر + المتوسطات المتحركة")
                st.line_chart(df[["Close","SMA_20","SMA_50"]].tail(150))