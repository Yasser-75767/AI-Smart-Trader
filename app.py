# app.py
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import datetime

st.title("AI Smart Trader 📈")

# ===== إعداد قائمة رموز أسهم قليلة السيولة =====
low_liquidity_symbols = ["GPRO", "BBBY", "JCPNQ", "MNKD", "PLUG"]

st.sidebar.header("إعدادات التطبيق")
symbol = st.sidebar.selectbox("اختر السهم:", low_liquidity_symbols)

start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())

if st.sidebar.button("تحميل البيانات وتحليلها"):
    st.write(f"🔹 جلب بيانات {symbol} من {start_date} إلى {end_date}")
    data = yf.download(symbol, start=start_date, end=end_date)
    
    if data.empty:
        st.warning("⚠️ لا توجد بيانات متاحة للفترة المختارة.")
    else:
        # إنشاء عمود الهدف
        data['Target'] = data['Close'].shift(-1)
        data = data.dropna()
        
        if data.empty:
            st.warning("⚠️ البيانات غير كافية لحساب التنبؤات.")
        else:
            st.write("📊 البيانات التاريخية:")
            st.dataframe(data.tail())
            
            # إعداد البيانات للتنبؤ
            X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            y = (data['Target'] > data['Close']).astype(int)
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # إنشاء النموذج وتدريبه
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
            
            # التنبؤ وتقييم النموذج
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            st.success(f"✅ دقة النموذج على بيانات الاختبار: {accuracy*100:.2f}%")
            
            st.write("📈 التوقعات الأخيرة:")
            results = pd.DataFrame({"Actual": y_test, "Prediction": preds})
            st.dataframe(results.tail())