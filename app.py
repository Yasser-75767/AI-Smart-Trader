# AI Smart Trader Pro — النسخة النهائية مع إشارات التداول
import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator
import streamlit as st
import datetime

# --- واجهة Streamlit ---
st.title("🎯 AI Smart Trader Pro — النسخة النهائية مع إشارات التداول")

# اختيار الأصل (رمز السهم)
symbol = st.selectbox("اختر الأصل (رمز السهم)", ["AAPL", "MSFT", "GOOGL", "AMZN"])

# اختيار التواريخ
start_date = st.date_input("تاريخ البداية", datetime.date(2020, 11, 28))
end_date = st.date_input("تاريخ النهاية", datetime.date(2025, 11, 28))

# إعدادات المؤشرات
min_lookback = st.number_input("أيام النظر للخلف (Min)", min_value=1, max_value=100, value=5)
max_lookback = st.number_input("أيام النظر للخلف (Max)", min_value=min_lookback, max_value=100, value=40)

confidence_min = st.number_input("حد الثقة لإشارة قوية (Min %)", min_value=0, max_value=100, value=0)
confidence_max = st.number_input("حد الثقة لإشارة قوية (Max %)", min_value=confidence_min, max_value=100, value=100)

# --- جلب البيانات ---
df = yf.download(symbol, start=start_date, end=end_date)

if df.empty:
    st.warning("لا توجد بيانات لهذا السهم أو للفترة المحددة.")
else:
    # تأكد أن العمود Close هو 1D
    df['Close'] = df['Close'].squeeze()

    # --- حساب المتوسطات ---
    try:
        df['SMA_5'] = SMAIndicator(df['Close'], window=5).sma_indicator()
        df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['EMA_10'] = EMAIndicator(df['Close'], window=10).ema_indicator()
    except Exception as e:
        st.error(f"خطأ في حساب المؤشرات: {e}")

    # --- تحضير الأعمدة للرسم ---
    columns_to_plot = [col for col in ['Close','SMA_5','SMA_20','EMA_10'] if col in df.columns]

    if columns_to_plot:
        st.subheader("📊 بيانات الأسعار والمتوسطات")
        st.line_chart(df[columns_to_plot].tail(150))
    else:
        st.warning("لا توجد أعمدة صحيحة للرسم البياني")