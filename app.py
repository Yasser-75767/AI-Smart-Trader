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

# إعداد الصفحة
st.set_page_config(page_title="AI Smart Trader", layout="wide")
st.title("AI Smart Trader 📈")

# -----------------------------
# قوائم الأسهم والفوركس
# -----------------------------
STOCKS_LIST = [
    "AAPL","MSFT","GOOG","AMZN","TSLA","NVDA","META","NFLX","BABA",
    "INTC","AMD","PYPL","ADBE","ORCL","CSCO"
]

FOREX_LIST = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X",
    "NZDUSD=X","USDCHF=X","EURJPY=X","EURGBP=X","EURCHF=X",
    "GBPJPY=X","AUDJPY=X","AUDNZD=X","CADJPY=X","CHFJPY=X"
]

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


# -----------------------------
# دالة تجيب أول رمز صالح (الرمز + بديل إن لزم)
# -----------------------------
def fetch_symbol_with_fallback(user_symbol: str, market_choice: str, days: int = 90):
    """
    تحاول استخدام الرمز الذي أدخله المستخدم،
    وإن لم توجد بيانات كافية له → تختار أفضل بديل تلقائياً.
    """
    if market_choice == "أسهم":
        fallback = STOCKS_LIST
    else:
        fallback = FOREX_LIST

    candidates = [user_symbol] + [s for s in fallback if s != user_symbol]

    for sym in candidates:
        try:
            data = yf.download(sym, period=f"{days}d")
        except Exception:
            continue

        if data is None or data.empty:
            continue

        # التأكد من وجود الأعمدة الأساسية
        if not all(col in data.columns for col in REQUIRED_COLS):
            continue

        # تنظيف القيم الناقصة
        data = data.dropna(subset=REQUIRED_COLS)
        if len(data) < 30:  # نريد بيانات كافية للتدريب
            continue

        return sym, data

    return None, None


# -----------------------------
# واجهة الإعدادات في الشريط الجانبي
# -----------------------------
st.sidebar.header("إعدادات التطبيق")
market_choice = st.sidebar.radio("اختر السوق:", ["أسهم", "فوركس"])

default_symbol = "AAPL" if market_choice == "أسهم" else "EURUSD=X"
symbol = st.sidebar.text_input(
    "أدخل رمز السهم أو زوج العملة (مثال: AAPL أو EURUSD=X):",
    value=default_symbol
)

# (اختياري) تواريخ، لكن نحن نستعمل آخر X يوم تلقائياً
start_date = st.sidebar.date_input("تاريخ البداية (للعرض فقط):", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("تاريخ النهاية (للعرض فقط):", datetime.date.today())

# تبويبات
tab_market, tab_image, tab_daily = st.tabs(["📊 بيانات السوق", "🖼️ تحليل الصور", "⭐ توصيات التداول اليومي"])


# ============================================
# 📊 تبويب بيانات السوق + نموذج التنبؤ
# ============================================
with tab_market:
    st.subheader("📊 تحليل بيانات السوق بالذكاء الاصطناعي")

    if st.button("تحميل البيانات وتحليلها"):
        if not symbol.strip():
            st.error("❌ الرجاء إدخال رمز أولاً.")
        else:
            used_symbol, data = fetch_symbol_with_fallback(symbol.strip(), market_choice, days=90)

            if data is None:
                st.error("⚠ لا توجد أي بيانات كافية لهذا الرمز ولا للبدائل حالياً.")
                if st.button("🔄 إعادة المحاولة", key="retry_market"):
                    st.session_state.clear()
                    st.rerun()
            else:
                if used_symbol != symbol.strip():
                    st.info(f"ℹ تم استخدام الرمز البديل: **{used_symbol}** لأن بيانات {symbol} غير كافية.")

                st.write("📊 آخر البيانات التاريخية:")
                st.dataframe(data.tail())

                # إنشاء الهدف: هل إغلاق الغد أعلى من اليوم؟
                data = data.copy()
                data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
                data = data.dropna(subset=REQUIRED_COLS + ["Target"])

                if data["Target"].nunique() < 2:
                    st.warning("⚠ البيانات لا تحتوي على صعود وهبوط كافيين لبناء نموذج.")
                else:
                    X = data[REQUIRED_COLS]
                    y = data["Target"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False
                    )

                    model = xgb.XGBClassifier(
                        use_label_encoder=False,
                        eval_metric="logloss"
                    )
                    model.fit(X_train, y_train)

                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)

                    st.success(f"✅ دقة النموذج على البيانات التاريخية: {acc*100:.2f}%")

                    # تنبؤ آخر شمعة
                    last_row = data.iloc[[-1]]  # على شكل DataFrame
                    last_pred = model.predict(last_row[REQUIRED_COLS])[0]

                    st.subheader("🔍 إشارة آخر فترة:")
                    if last_pred == 1:
                        st.success(f"📈 إشارة محتملة: صعود ({used_symbol}) - يمكن التفكير في الشراء بحذر.")
                    else:
                        st.error(f"📉 إشارة محتملة: هبوط ({used_symbol}) - الحذر من الشراء.")

    # زر إعادة المحاولة في هذا التبويب
    if st.button("🔄 إعادة تحميل الصفحة", key="market_full_retry"):
        st.session_state.clear()
        st.rerun()


# ============================================
# 🖼️ تبويب تحليل الصور (لقطات الشاشة)
# ============================================
with tab_image:
    st.subheader("🖼️ تحليل صور الشموع والمنحنيات من لقطات الشاشة")

    uploaded = st.file_uploader("📤 ارفعي صورة من هاتفك (شموع يابانية أو منحنى سعري):", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="الصورة المرفوعة", use_column_width=True)

        # تحويل إلى OpenCV
        img_rgb = np.array(image.convert("RGB"))
        img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        white_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        ratio = white_pixels / total_pixels if total_pixels > 0 else 0

        st.write(f"نسبة الحواف في الصورة: {ratio*100:.2f}%")

        # تحليل بدائي بسيط للاتجاه
        if ratio > 0.06:
            st.success("📈 الصورة توحي بحركة سوق نشيطة (اتجاه محتمل صاعد أو متقلب بقوة).")
        else:
            st.warning("📉 الصورة توحي بحركة ضعيفة أو اتجاه هابط/هادئ.")

        st.image(edges, caption="نتيجة تحليل الحواف", use_column_width=True)

    if st.button("🔄 إعادة المحاولة (الصور)", key="img_retry"):
        st.session_state.clear()
        st.rerun()


# ============================================
# ⭐ تبويب توصيات التداول اليومي
# ============================================
with tab_daily:
    st.subheader("⭐ توصيات تداول يومية سريعة")

    if st.button("احسب التوصيات الآن"):
        symbols_source = STOCKS_LIST if market_choice == "أسهم" else FOREX_LIST
        good_symbols = []

        for sym in symbols_source:
            try:
                df = yf.download(sym, period="5d")
            except Exception:
                continue

            if df is None or df.empty or len(df) < 2:
                continue

            if not all(col in df.columns for col in ["Open", "Close"]):
                continue

            df = df.dropna(subset=["Open", "Close"])
            if df.empty:
                continue

            last = df.iloc[-1]
            close_val = float(last["Close"])
            open_val = float(last["Open"])

            if close_val > open_val:
                good_symbols.append(sym)
        if not good_symbols:
            st.warning("⚠ لم يتم العثور على رموز قوية اليوم بناءً على بيانات الأيام الأخيرة.")
        else:
            st.success("✅ هذه أفضل الرموز التي أغلقت أعلى من الافتتاح في آخر جلسة:")
            st.write(", ".join(good_symbols))

    if st.button("🔄 إعادة المحاولة (التوصيات)", key="daily_retry"):
        st.session_state.clear()
        st.rerun()