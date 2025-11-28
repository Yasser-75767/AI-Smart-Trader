# AI Smart Trader Pro — النسخة النهائية بدون أخطاء 1D vs 2D
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import streamlit as st

st.title("🎯 AI Smart Trader Pro — النسخة النهائية مع إشارات التداول")

# ---- إعدادات المستخدم ----
symbol = st.selectbox("اختر الأصل:", ["AAPL", "GOOGL", "MSFT", "TSLA"])
start_date = st.date_input("تاريخ البداية", pd.to_datetime("2020-11-28"))
end_date = st.date_input("تاريخ النهاية", pd.to_datetime("2025-11-28"))
min_back = st.number_input("أيام النظر للخلف (Min)", min_value=1, value=5)
max_back = st.number_input("أيام النظر للخلف (Max)", min_value=min_back, value=40)
confidence_min = st.number_input("حد الثقة لإشارة قوية (Min %)", min_value=0, max_value=100, value=0)
confidence_max = st.number_input("حد الثقة لإشارة قوية (Max %)", min_value=confidence_min, max_value=100, value=100)

# ---- تحميل البيانات ----
@st.cache_data
def load_data(sym, start, end):
    df = yf.download(sym, start=start, end=end)
    df = df.reset_index()
    return df

df = load_data(symbol, start_date, end_date)

# ---- حساب المؤشرات ----
try:
    df['SMA_5'] = SMAIndicator(df['Close'].values.flatten(), window=5).sma_indicator()
    df['SMA_20'] = SMAIndicator(df['Close'].values.flatten(), window=20).sma_indicator()
    df['EMA_10'] = EMAIndicator(df['Close'].values.flatten(), window=10).ema_indicator()
    
    macd = MACD(df['Close'].values.flatten())
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    df['RSI'] = RSIIndicator(df['Close'].values.flatten()).rsi()
    
    df['Volume_SMA'] = SMAIndicator(df['Volume'].values.flatten(), window=20).sma_indicator()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].replace(0, np.nan)
    
    indicators_error = None
except Exception as e:
    indicators_error = str(e)

# ---- زر الحصول على النتائج ----
if st.button("📊 الحصول على النتائج"):
    if indicators_error:
        st.error(f"خطأ في حساب المؤشرات: {indicators_error}")
    else:
        # ---- عرض البيانات ----
        st.subheader("📈 بيانات الأسعار والمؤشرات")
        columns_to_plot = ['Close','SMA_5','SMA_20','EMA_10']
        existing_columns = [col for col in columns_to_plot if col in df.columns]
        if existing_columns:
            st.line_chart(df[existing_columns].tail(150))
        else:
            st.warning("لا توجد أعمدة للرسم بعد.")

        # ---- إشارات تداول بسيطة ----
        st.subheader("🎯 إشارات التداول")
        signals = []
        for i in range(len(df)):
            if pd.notna(df['MACD'].iloc[i]) and pd.notna(df['MACD_signal'].iloc[i]):
                if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                    signals.append("شراء")
                elif df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                    signals.append("بيع")
                else:
                    signals.append("محايد")
            else:
                signals.append("غير متاح")
        df['Signal'] = signals
        st.dataframe(df[['Date','Close','Signal']].tail(20))