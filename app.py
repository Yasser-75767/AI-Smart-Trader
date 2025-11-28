import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.set_page_config(page_title="AI Smart Trader Pro", layout="wide")

st.title("🎯 AI Smart Trader Pro — النسخة النهائية مع إشارات التداول")

# --- إدخال المستخدم ---
symbol = st.text_input("اختر الأصل (رمز السهم)", value="AAPL")
start_date = st.date_input("تاريخ البداية", pd.to_datetime("2020-11-28"))
end_date = st.date_input("تاريخ النهاية", pd.to_datetime("2025-11-28"))
lookback_min = st.number_input("أيام النظر للخلف (Min)", min_value=1, value=5)
lookback_max = st.number_input("أيام النظر للخلف (Max)", min_value=1, value=40)
conf_min = st.number_input("حد الثقة لإشارة قوية (Min %)", min_value=0, max_value=100, value=0)
conf_max = st.number_input("حد الثقة لإشارة قوية (Max %)", min_value=0, max_value=100, value=100)

# زر لتحليل البيانات
if st.button("الحصول على النتائج ✅"):

    # --- تحميل البيانات ---
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        st.error("لا توجد بيانات للسهم المحدد.")
    else:
        df = df.copy()

        # --- التأكد من 1D لكل عمود ---
        close = df['Close'].squeeze()
        volume = df['Volume'].squeeze()

        # --- حساب المؤشرات ---
        try:
            df['SMA_5'] = SMAIndicator(close, window=5).sma_indicator()
            df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
            df['EMA_10'] = EMAIndicator(close, window=10).ema_indicator()
        except Exception as e:
            st.warning(f"خطأ في حساب المتوسطات: {e}")

        try:
            macd = MACD(close)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
        except Exception as e:
            st.warning(f"خطأ في حساب MACD: {e}")

        try:
            df['RSI'] = RSIIndicator(close, window=14).rsi()
        except Exception as e:
            st.warning(f"خطأ في حساب RSI: {e}")

        try:
            df['Volume_SMA'] = SMAIndicator(volume, window=20).sma_indicator()
            df['Volume_Ratio'] = volume / df['Volume_SMA']
        except Exception as e:
            st.warning(f"خطأ في حساب Volume Ratio: {e}")

        # --- عرض البيانات ---
        st.subheader("📊 بيانات الأسعار والمؤشرات")
        st.dataframe(df.tail(10))

        # --- اختيار الأعمدة للعرض ---
        columns_to_plot = [col for col in ['Close','SMA_5','SMA_20','EMA_10'] if col in df.columns]
        if columns_to_plot:
            st.line_chart(df[columns_to_plot].tail(150))

        # --- إشارات شراء/بيع ---
        st.subheader("📈 إشارات التداول")
        signals = []
        for i in range(len(df)):
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                    signals.append("شراء")
                elif df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                    signals.append("بيع")
                else:
                    signals.append("محايد")
            else:
                signals.append("غير متوفر")
        df['Signal'] = signals
        st.dataframe(df[['Close','Signal']].tail(10))