import yfinance as yf
import pandas as pd
import streamlit as st

st.title("🎯 AI Smart Trader Pro — نسخة أبسط")

# اختيار السهم
symbol = st.selectbox("اختر الأصل (رمز السهم)", ["AAPL","MSFT","GOOGL"])
start_date = st.date_input("تاريخ البداية", pd.to_datetime("2020-11-28"))
end_date = st.date_input("تاريخ النهاية", pd.to_datetime("2025-11-28"))

# تحميل البيانات
df = yf.download(symbol, start=start_date, end=end_date)

if df.empty:
    st.error("لا توجد بيانات لهذا الرمز.")
else:
    # فقط العمود Close
    df['Close'] = df['Close'].values.flatten()

    # إشارات شراء/بيع عشوائية للعرض فقط
    import numpy as np
    df['Signal'] = np.random.choice([-1,0,1], size=len(df))

    # رسم خط Close فقط
    st.subheader("📈 أسعار الإغلاق")
    st.line_chart(df['Close'])

    # عرض آخر 20 صف مع الإشارة
    st.subheader("💹 إشارات التداول (عرض تجريبي)")
    st.write(df[['Close','Signal']].tail(20))