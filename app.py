# app.py — AI Smart Trader النهائية (سريعة + RSI/MACD + إيميل) 💜
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import datetime
import random
import smtplib
import ssl

# ===== إعداد الصفحة =====
st.set_page_config(page_title="AI Smart Trader — النسخة المطورة 💜", layout="wide")

# ===== الرموز =====
stock_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
forex_symbols = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
all_symbols = stock_symbols + forex_symbols

# ===== أعمدة الميزات الثابتة (C + مؤشرات إضافية) =====
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Price_Range", "Price_Change",
    "MA_5", "Volume_MA",
    "RSI_14", "MACD", "MACD_Signal"
]

# ============================================================================
# دوال مساعدة
# ============================================================================

@st.cache_data(show_spinner=False)
def fetch_data(symbol, start, end):
    """تحميل البيانات من ياهو (مع كاش لتسريع التطبيق)."""
    return yf.download(symbol, start=start, end=end, progress=False)

def load_data_with_fallback(original_symbol, start, end):
    """يحاول تحميل الرمز المختار، وإن فشل يجرب بدائل."""
    candidates = [original_symbol] + [s for s in all_symbols if s != original_symbol]

    for sym in candidates:
        try:
            df = fetch_data(sym, start, end)
        except Exception:
            continue

        base_cols = ["Open", "High", "Low", "Close", "Volume"]
        if df.empty or not all(c in df.columns for c in base_cols):
            continue

        df = df[base_cols].dropna()
        if len(df) < 25:
            continue

        if sym != original_symbol:
            st.info(f"ℹ تم استخدام الرمز البديل: {sym} بدل {original_symbol}")
        return df, sym

    return pd.DataFrame(), original_symbol

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def prepare_features(df, with_target=True):
    """نفس تجهيز الميزات للتدريب والتنبؤ (ثابت)."""
    df = df.copy()

    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in base_cols):
        return None, None, None

    # الهدف
    if with_target:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # الميزات الكلاسيكية
    df["Price_Range"] = df["High"] - df["Low"]
    df["Price_Change"] = df["Close"] - df["Open"]
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["Volume_MA"] = df["Volume"].rolling(window=5).mean()

    # RSI + MACD
    df["RSI_14"] = compute_rsi(df["Close"], period=14)
    macd, macd_sig = compute_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_Signal"] = macd_sig

    # ضمان وجود كل الأعمدة وملء الفراغات
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    if with_target:
        # لو صار كله NaN في Target → نعتبره 0 بدل ما ينهار
        if "Target" not in df.columns:
            df["Target"] = 0
        df = df.dropna(subset=["Target"])
        if df.empty:
            return None, None, None

        X = df[FEATURE_COLS]
        y = df["Target"].astype(int)
        return X, y, df
    else:
        X = df[FEATURE_COLS]
        return X, None, df

def train_model(df):
    """تدريب نموذج XGBoost مع مؤشرات فنية."""
    X, y, df_feat = prepare_features(df, with_target=True)
    if X is None or y is None:
        st.warning("⚠ البيانات غير كافية لتجهيز الميزات والهدف.")
        return None, None

    if len(X) < 40:
        st.warning("⚠ البيانات أقل من 40 نقطة، النموذج قد لا يكون دقيقاً.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    try:
        model = xgb.XGBClassifier(
            n_estimators=90,
            max_depth=4,
            learning_rate=0.08,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return model, acc
    except Exception as e:
        st.error(f"⚠ خطأ في تدريب النموذج: {e}")
        return None, None

def predict_last(model, df):
    """التنبؤ باتجاه آخر شمعة في البيانات."""
    X_pred, _, df_clean = prepare_features(df, with_target=False)
    if X_pred is None or X_pred.empty:
        st.warning("⚠ لا توجد بيانات كافية للتنبؤ.")
        return None

    last_row = X_pred.iloc[[-1]].values  # (1, n_features)
    try:
        return model.predict(last_row)[0]
    except Exception as e:
        st.error(f"⚠ خطأ أثناء التوقع: {e}")
        return None

def analyze_image(file):
    """تحليل بسيط لصورة الشموع (تجريبي)."""
    try:
        image = Image.open(file).convert("RGB")
        image = image.resize((256, 256))
        st.image(image, caption="📊 الصورة المحملة", use_column_width=True)

        img_cv = np.array(image)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        mean_val = float(np.mean(img_gray))
        st.write(f"📊 متوسط الإضاءة: {mean_val:.1f}")

        return 1 if mean_val > 120 else 0
    except Exception as e:
        st.error(f"⚠ خطأ في تحليل الصورة: {e}")
        return None

def send_email_alert(smtp_server, smtp_port, email_from, email_pass, email_to, subject, body):
    """إرسال تنبيه عبر الإيميل (اختياري)."""
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(email_from, email_pass)
            msg = f"Subject: {subject}\n\n{body}"
            server.sendmail(email_from, email_to, msg)
        st.success("📧 تم إرسال تنبيه عبر البريد الإلكتروني.")
    except Exception as e:
        st.warning(f"⚠ لم يتم إرسال البريد: {e}")

# ============================================================================
# واجهة المستخدم
# ============================================================================

st.title("📈 AI Smart Trader — النسخة المطورة 💜")
st.caption("تطبيق تعليمي لتجربة الذكاء الاصطناعي في تحليل الأسهم و الفوركس.")

st.warning("⚠ التوقعات تعليمية فقط، التداول الحقيقي يحمل مخاطر مالية.")

# ===== خيارات إضافية في الشريط الجانبي =====
st.sidebar.markdown("---")
st.sidebar.subheader("تنبيهات البريد الإلكتروني (اختياري)")
enable_email = st.sidebar.checkbox("تفعيل تنبيه عبر الإيميل عند إشارة شراء")

smtp_server = st.sidebar.text_input("SMTP Server (مثال: smtp.gmail.com)", value="", help="للاستخدام المتقدم فقط")
smtp_port = st.sidebar.number_input("SMTP Port", value=465, step=1)
email_from = st.sidebar.text_input("بريد المرسل (حسابك)", value="")
email_pass = st.sidebar.text_input("كلمة مرور التطبيق", type="password", value="")
email_to = st.sidebar.text_input("بريد المستلم", value="")

# ===== Tabs =====
tab1, tab2, tab3 = st.tabs(["📊 التوقع والتحليل", "📷 تحليل الصور", "ℹ️ معلومات إضافية"])

# ----------------------------------------------------------------------------
# تبويب 1: التوقع والتحليل
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("📊 توقع اتجاه السوق")

    if st.button("🚀 الحصول على التوصيات", key="predict_button"):
        with st.spinner("⏳ جاري تحميل البيانات وتحليلها..."):
            df, used_symbol = load_data_with_fallback(symbol, start_date, end_date)
            if df.empty:
                st.error("⚠ لا توجد بيانات كافية لهذا الرمز أو البدائل.")
                st.stop()

            model, acc = train_model(df)
            if model is None:
                st.stop()

            pred = predict_last(model, df)
            if pred is None:
                st.stop()

            st.success(f"✔ دقة النموذج على البيانات التاريخية: {acc*100:.2f}%")

            if pred == 1:
                st.success(f"🔥 التوقع: {used_symbol} في اتجاه صاعد (إشارة شراء تعليمية)")
                # إرسال إيميل إن تم تفعيله
                if enable_email:
                    if all([smtp_server, email_from, email_pass, email_to]):
                        body = f"إشارة شراء تعليمية من AI Smart Trader لرمز: {used_symbol}"
                        send_email_alert(
                            smtp_server, smtp_port,
                            email_from, email_pass,
                            email_to,
                            subject=f"إشارة تعليمية: شراء {used_symbol}",
                            body=body
                        )
                    else:
                        st.info("ℹ لتفعيل الإيميل، رجاءً املئي جميع حقول الإيميل في الشريط الجانبي.")
            else:
                st.warning(f"📉 التوقع: {used_symbol} في اتجاه هابط أو ضعيف (تجنب الشراء)")

            st.markdown("### 🔎 آخر البيانات التاريخية:")
            st.dataframe(df.tail(15))

            # إحصائيات سريعة
            st.markdown("### 📈 إحصائيات أساسية")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("متوسط الإغلاق", f"{df['Close'].mean():.2f}")
            with col2:
                st.metric("أعلى سعر", f"{df['High'].max():.2f}")
            with col3:
                st.metric("أقل سعر", f"{df['Low'].min():.2f}")
            with col4:
                st.metric("متوسط الحجم", f"{df['Volume'].mean():.0f}")

# ----------------------------------------------------------------------------
# تبويب 2: تحليل الصور
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("📷 تحليل صورة الشموع / المنحنى")
    if uploaded_file is None:
        st.info("📎 من الشريط الجانبي، ارفعي صورة للشارت (ScreenShot) لتحليلها.")
    else:
        pred_img = analyze_image(uploaded_file)
        if pred_img == 1:
            st.success("🔥 استنادًا إلى إضاءة الصورة: السوق يبدو صاعدًا (تحليل تجريبي فقط).")
        elif pred_img == 0:
            st.warning("📉 استنادًا إلى إضاءة الصورة: السوق يبدو هابطًا أو ضعيفًا (تحليل تجريبي فقط).")
        else:
            st.info("⚠ لم يتمكن التطبيق من تحليل الصورة.")

# ----------------------------------------------------------------------------
# تبويب 3: معلومات إضافية
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("ℹ️ معلومات عن التطبيق")
    st.write("""
    - هذا التطبيق تعليمي فقط، الهدف منه تدريبك على:
      - تحميل بيانات الأسهم والفوركس.
      - تجربة نموذج XGBoost مع ميزات فنية (RSI, MACD, MA...).
      - الحصول على توقع اتجاه (صعود/هبوط) بشكل مبسط.
      - تجربة تحليل بسيط للصور.
    - لا يُستخدم هذا التطبيق لاتخاذ قرارات استثمارية حقيقية.
    """)

    st.markdown("### ⭐ رموز مقترحة للمراقبة (تعليميًا)")
    st.write(random.sample(all_symbols, 5))