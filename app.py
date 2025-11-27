import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import datetime

st.title("🔮 AI Smart Trader - توصيات تداول يومية")

# -------------------------------------------------
# وظيفة تحميل آمنة
# -------------------------------------------------
def load_data_safe(symbol):
    try:
        # تحميل آخر 60 يوم تلقائيًا
        data = yf.download(symbol, period="60d")

        # التحقق من وجود الأعمدة
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                return None

        # إزالة القيم الناقصة
        data = data.dropna(subset=required_cols)

        if data.empty:
            return None

        return data
    except:
        return None


# -------------------------------------------------
# واجهة التطبيق
# -------------------------------------------------

symbol = st.text_input("أدخل رمز السهم أو العملة (مثال: AAPL أو EURUSD=X):")

if st.button("📥 تحميل البيانات"):
    if not symbol.strip():
        st.error("❌ الرجاء إدخال رمز صالح.")
    else:
        df = load_data_safe(symbol)

        if df is None:
            st.error("⚠ لا توجد بيانات كافية لهذا الرمز. جرّب رمزًا آخر.")
        else:
            st.success("✅ تم تحميل البيانات بنجاح!")
            st.dataframe(df.tail())


# -------------------------------------------------
# زر التوصيات
# -------------------------------------------------

if st.button("📊 الحصول على التوصيات"):
    if not symbol.strip():
        st.error("❌ الرجاء إدخال رمز أولاً.")
    else:
        data = load_data_safe(symbol)

        if data is None or len(data) < 10:
            st.warning("⚠ البيانات غير كافية لعمل التنبؤ. جرّب رمزًا آخر.")
        else:
            # إعداد الهدف
            data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
            data = data.dropna()

            # تدريب النموذج
            X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            y = data['Target']

            if y.nunique() < 2:
                st.warning("⚠ البيانات غير كافية للتنبؤ (كل القيم متشابهة).")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                model = xgb.XGBClassifier()
                model.fit(X_train, y_train)

                accuracy = accuracy_score(y_test, model.predict(X_test))

                # آخر صف
                last_row = data.iloc[-1:]

                prediction = model.predict(last_row[['Open','High','Low','Close','Volume']])[0]

                st.subheader("🔍 دقة النموذج: {:.2f}%".format(accuracy * 100))

                if prediction == 1:
                    st.success(f"📈 توصية: شراء {symbol}")
                else:
                    st.error(f"📉 توصية: بيع {symbol}")

# -------------------------------------------------
# إعادة المحاولة
# -------------------------------------------------

if st.button("🔄 إعادة المحاولة"):
    st.experimental_rerun()