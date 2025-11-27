# app.py
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import datetime
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="AI Smart Trader", layout="wide")
st.title("AI Smart Trader 📈")

# ===== قوائم الأسهم والفوركس =====
stocks_symbols = [
    "AAPL","MSFT","GOOG","AMZN","TSLA","FB","NVDA","NFLX","BABA",
    "INTC","AMD","PYPL","ADBE","ORCL","CSCO"
]

forex_symbols = [
    "EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X",
    "NZDUSD=X","USDCHF=X","EURJPY=X","EURGBP=X","EURCHF=X",
    "GBPJPY=X","AUDJPY=X","AUDNZD=X","CADJPY=X","CHFJPY=X"
]

# ===== واجهة المستخدم =====
st.sidebar.header("إعدادات التطبيق")
market_choice = st.sidebar.radio("اختر السوق:", ["أسهم","فوركس"])
symbol = st.sidebar.selectbox("اختر السهم أو زوج العملات:",
                              stocks_symbols if market_choice=="أسهم" else forex_symbols)
start_date = st.sidebar.date_input("تاريخ البداية:", datetime.date(2022,1,1))
end_date = st.sidebar.date_input("تاريخ النهاية:", datetime.date.today())

# ===== تبويبات التطبيق =====
tabs = st.tabs(["📊 بيانات السوق", "🖼️ تحليل الصور", "⭐ توصيات التداول اليومي"])

# ===== دالة تحميل البيانات بأمان =====
def load_data_safe(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        required_cols = ['Open','High','Low','Close','Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if data.empty or missing_cols:
            return None, missing_cols
        data['Target'] = data['Close'].shift(-1)
        data = data.dropna(subset=required_cols + ['Target'])
        if len(data) < 2:
            return None, None
        return data, None
    except Exception as e:
        return None, None

# ===== تبويب بيانات السوق =====
with tabs[0]:
    if st.button("تحميل البيانات وتحليلها"):
        data, missing_cols = load_data_safe(symbol, start_date, end_date)
        if data is None:
            msg = "⚠️ البيانات غير كافية للتنبؤ."
            if missing_cols:
                msg += f" الأعمدة المفقودة: {', '.join(missing_cols)}"
            st.warning(msg)
            if st.button("🔄 إعادة المحاولة"):
                st.experimental_rerun()
        else:
            st.write("📊 البيانات التاريخية:")
            st.dataframe(data.tail())

            X = data[['Open','High','Low','Close','Volume']]
            y = (data['Target'] > data['Close']).astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train,y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test,preds)
            st.success(f"✅ دقة النموذج على بيانات الاختبار: {accuracy*100:.2f}%")

            st.write("📈 التوقعات الأخيرة:")
            results = pd.DataFrame({"Actual": y_test, "Prediction": preds})
            st.dataframe(results.tail())

# ===== تبويب تحليل الصور =====
with tabs[1]:
    st.write("📤 قم برفع صورة الشموع اليابانية أو الرسوم البيانية")
    uploaded_file = st.file_uploader("اختر صورة من هاتفك أو الكمبيوتر", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة المرفوعة", use_column_width=True)
        img_array = np.array(image.convert("RGB"))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        edges = cv2.Canny(blur, threshold1=50, threshold2=150)
        white_pixels = np.sum(edges>0)
        total_pixels = edges.size
        white_ratio = white_pixels/total_pixels
        if white_ratio > 0.05:
            st.success("📈 اتجاه السوق محتمل أن يكون صاعد")
        else:
            st.error("📉 اتجاه السوق محتمل أن يكون هابط")
        st.image(edges, caption="صورة بعد تحليل الحواف", use_column_width=True)

# ===== تبويب توصيات التداول اليومي =====
with tabs[2]:
    st.write("⭐ أفضل الأسهم أو أزواج الفوركس للتداول اليومي بناءً على بيانات اليوم")
    if st.button("احسب التوصيات"):
        today_data = []
        symbols_to_check = stocks_symbols if market_choice=="أسهم" else forex_symbols
        for sym in symbols_to_check:
            try:
                df = yf.download(sym, period="5d")
                if df.empty or len(df)<2:
                    continue
                last_row = df.iloc[-1]
                if pd.notnull(last_row.get('Close')) and pd.notnull(last_row.get('Open')):
                    if float(last_row['Close']) > float(last_row['Open']):
                        today_data.append(sym)
            except Exception:
                continue
        if len(today_data)==0:
            st.warning("⚠️ لم يتم العثور على أي توصيات اليوم.")
            if st.button("🔄 إعادة المحاولة"):
                st.experimental_rerun()
        else:
            st.success(f"أفضل الخيارات اليوم: {', '.join(today_data)}")