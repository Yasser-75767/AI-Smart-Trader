# AI Smart Trader Pro — النسخة النهائية مع إشارات التداول
import yfinance as yf
import pandas as pd
import streamlit as st
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.title("🎯 AI Smart Trader Pro — النسخة النهائية")

# --- واجهة المستخدم ---
symbol = st.text_input("اختر الأصل (رمز السهم)", "AAPL")
start_date = st.date_input("تاريخ البداية", pd.to_datetime("2020-11-28"))
end_date = st.date_input("تاريخ النهاية", pd.to_datetime("2025-11-28"))
min_lookback = st.number_input("أيام النظر للخلف (Min)", min_value=1, value=5)
max_lookback = st.number_input("أيام النظر للخلف (Max)", min_value=min_lookback, value=40)
conf_min = st.number_input("حد الثقة لإشارة قوية (Min %)", min_value=0, max_value=100, value=12)
conf_max = st.number_input("حد الثقة لإشارة قوية (Max %)", min_value=conf_min, max_value=100, value=100)

# --- زر الحصول على النتائج ---
if st.button("📊 الحصول على النتائج"):

    # --- تحميل البيانات ---
    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        st.warning("لا توجد بيانات لهذا السهم خلال الفترة المحددة")
    else:
        # --- تحويل الأعمدة إلى 1D للتأكد من عدم حدوث أخطاء ---
        df['Close'] = df['Close'].squeeze()
        df['Volume'] = df['Volume'].squeeze()

        # --- حساب المؤشرات ---
        try:
            df['SMA_5'] = SMAIndicator(df['Close'], window=5).sma_indicator()
            df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
            df['EMA_10'] = EMAIndicator(df['Close'], window=10).ema_indicator()
            macd = MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['RSI'] = RSIIndicator(df['Close']).rsi()
        except Exception as e:
            st.error(f"خطأ في حساب المؤشرات: {e}")

        # --- تحديد الأعمدة المتاحة للرسم ---
        columns_to_plot = [col for col in ['Close','SMA_5','SMA_20','EMA_10'] if col in df.columns]

        if columns_to_plot:
            st.subheader("📈 بيانات الأسعار والمتوسطات")
            st.line_chart(df[columns_to_plot].tail(150))
        else:
            st.warning("لا توجد أعمدة صحيحة للرسم البياني")

        # --- عرض بيانات الإغلاق والمؤشرات ---
        st.subheader("📝 معاينة البيانات")
        st.dataframe(df.tail(10))