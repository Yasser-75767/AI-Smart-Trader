# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime

# ===== إعداد الصفحة =====
st.set_page_config(
    page_title="AI Smart Trader — نسخة الهاتف 💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 AI Smart Trader — النسخة الخفيفة للهاتف 💎")
st.warning("⚠ هذه أداة تعليمية فقط، التداول الحقيقي يحمل مخاطر مالية")

# ===== الإعدادات =====
symbol = st.selectbox("اختر سهم:", ["AAPL","MSFT","GOOGL","NVDA","AMZN"])
start_date = st.date_input("تاريخ البداية:", datetime.date(2020,1,1))
end_date = st.date_input("تاريخ النهاية:", datetime.date.today())

uploaded_file = st.file_uploader("📷 رفع صورة (اختياري)", type=["png","jpg","jpeg"])

# ===== دوال التحليل =====
def load_data(symbol):
    """
    تحميل البيانات من ملف CSV موجود مسبقًا في المستودع
    """
    try:
        df = pd.read_csv(f"{symbol}.csv", parse_dates=['Date'])
        df = df[df['Date'] >= pd.to_datetime(start_date)]
        df = df[df['Date'] <= pd.to_datetime(end_date)]
        if df.empty:
            st.error("❌ لا توجد بيانات لهذا النطاق الزمني")
            return None
        return df
    except Exception as e:
        st.error(f"❌ خطأ في تحميل البيانات: {e}")
        return None

def calculate_indicators(df):
    """
    حساب مؤشرات بسيطة وخفيفة
    """
    df = df.copy()
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Range'] = df['High'] - df['Low']
    return df

def predict_trend(df):
    """
    التنبؤ بالاتجاه بناءً على المتوسطات المتحركة
    """
    last = df.iloc[-1]
    if last['MA_5'] > last['MA_20']:
        return "📈 صاعد"
    else:
        return "📉 هابط"

def analyze_image(file):
    """
    تحليل بسيط جدًا للصور (مقياس الإضاءة)
    """
    from PIL import Image, ImageStat
    try:
        image = Image.open(file).convert('L')  # تحويل للصورة رمادية
        stat = ImageStat.Stat(image)
        mean_brightness = stat.mean[0]
        if mean_brightness > 120:
            return "📈 صاعد (الصورة مضيئة)"
        else:
            return "📉 هابط (الصورة مظلمة)"
    except:
        return "⚠ لا يمكن تحليل الصورة"

# ===== تنفيذ التحليل =====
if st.button("🚀 بدء التحليل"):
    df = load_data(symbol)
    if df is not None:
        df = calculate_indicators(df)
        trend = predict_trend(df)
        
        st.write("### 📊 الإحصائيات الأساسية:")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("متوسط الإغلاق", f"{df['Close'].mean():.2f}")
        with col2: st.metric("أعلى سعر", f"{df['High'].max():.2f}")
        with col3: st.metric("أقل سعر", f"{df['Low'].min():.2f}")
        
        st.write("### 📈 التنبؤ بالاتجاه:")
        st.success(f"**الاتجاه المتوقع: {trend}**")
        
        st.write("### 📊 آخر 10 أيام تداول:")
        st.dataframe(df.tail(10))
        st.line_chart(df['Close'].tail(100))
        
        if uploaded_file is not None:
            image_result = analyze_image(uploaded_file)
            st.write("### 📷 تحليل الصورة:")
            st.info(image_result)

st.markdown("---")
st.info("📝 ملاحظات مهمة: هذه أداة تعليمية فقط، لا تعتمد على التنبؤات للتداول الحقيقي.")