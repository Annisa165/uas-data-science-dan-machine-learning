import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Performa Akademik",
    page_icon="🎓",
    layout="centered"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 0.4rem 0; }
    .hero p  { font-size: 0.95rem; opacity: 0.85; margin: 0; }

    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }
    .card h3 {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin: 0 0 1rem 0;
    }
    .result-high {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
    }
    .result-medium {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border-left: 5px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
    }
    .result-low {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
    }
    .result-label { font-size: 1.6rem; font-weight: 700; margin: 0 0 0.3rem 0; }
    .result-desc  { font-size: 0.9rem; opacity: 0.8; margin: 0; }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model_rf.pkl",      "rb") as f: rf_model  = pickle.load(f)
    with open("model_svm.pkl",     "rb") as f: svm_model = pickle.load(f)
    with open("model_knn.pkl",     "rb") as f: knn_model = pickle.load(f)
    with open("scaler.pkl",        "rb") as f: scaler    = pickle.load(f)
    with open("imputer.pkl",       "rb") as f: imputer   = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f: le        = pickle.load(f)
    return rf_model, svm_model, knn_model, scaler, imputer, le

try:
    rf_model, svm_model, knn_model, scaler, imputer, label_encoder = load_artifacts()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    missing_file = str(e)

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎓 Prediksi Performa Akademik Siswa</h1>
    <p>Masukkan data kebiasaan belajar dan gaya hidup siswa untuk memprediksi performa akademiknya menggunakan Random Forest, SVM, dan KNN.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ File model tidak ditemukan: {missing_file}")
    st.info("Pastikan semua file .pkl ada di folder yang sama dengan app.py:\nmodel_rf.pkl, model_svm.pkl, model_knn.pkl, scaler.pkl, imputer.pkl, label_encoder.pkl")
    st.stop()

# ─────────────────────────────────────────
# PILIH MODEL
# ─────────────────────────────────────────
st.markdown('<div class="card"><h3>⚙️ Pilih Algoritma</h3>', unsafe_allow_html=True)
algoritma = st.radio(
    "Algoritma yang digunakan untuk prediksi:",
    ["Random Forest", "SVM", "KNN"],
    horizontal=True
)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FORM INPUT
# ─────────────────────────────────────────
st.markdown('<div class="card"><h3>📋 Data Siswa</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age               = st.number_input("Usia (tahun)", min_value=10, max_value=30, value=17)
    study_hours       = st.slider("Jam belajar per hari", 0.0, 12.0, 4.0, step=0.5)
    attendance        = st.slider("Tingkat kehadiran (%)", 0, 100, 80)
    sleep_hours       = st.slider("Jam tidur per hari", 0.0, 12.0, 7.0, step=0.5)
with col2:
    physical_activity = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 20.0, 3.0, step=0.5)
    mental_health     = st.slider("Skor kesehatan mental (1–10)", 1, 10, 7)
    social_activities = st.slider("Aktivitas sosial (jam/minggu)", 0.0, 20.0, 5.0, step=0.5)
    gpa               = st.number_input("GPA saat ini", min_value=0.0, max_value=4.0, value=3.0, step=0.01)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="card"><h3>📌 Data Tambahan</h3>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    gender          = st.selectbox("Jenis kelamin", ["Male", "Female"])
    part_time_job   = st.selectbox("Punya pekerjaan paruh waktu?", ["Yes", "No"])
    extracurricular = st.selectbox("Aktif ekstrakurikuler?", ["Yes", "No"])
with col4:
    internet_quality = st.selectbox("Kualitas internet", ["Poor", "Average", "Good"])
    family_income    = st.selectbox("Pendapatan keluarga", ["Low", "Medium", "High"])
    learning_style   = st.selectbox("Gaya belajar", ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"])

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang"):

    # Susun input sesuai urutan kolom saat training
    FEATURES = [
        "Age", "Study_Hours_Per_Day", "Attendance_Rate",
        "Sleep_Hours_Per_Day", "Physical_Activity_Hours_Per_Week",
        "Mental_Health_Score", "Social_Activities_Hours_Per_Week", "GPA",
        "Gender", "Part_Time_Job", "Extracurricular_Activities",
        "Internet_Quality", "Family_Income_Level", "Learning_Style"
    ]

    input_raw = pd.DataFrame([{
        "Age"                              : age,
        "Study_Hours_Per_Day"              : study_hours,
        "Attendance_Rate"                  : attendance,
        "Sleep_Hours_Per_Day"              : sleep_hours,
        "Physical_Activity_Hours_Per_Week" : physical_activity,
        "Mental_Health_Score"              : mental_health,
        "Social_Activities_Hours_Per_Week" : social_activities,
        "GPA"                              : gpa,
        "Gender"                           : gender,
        "Part_Time_Job"                    : part_time_job,
        "Extracurricular_Activities"       : extracurricular,
        "Internet_Quality"                 : internet_quality,
        "Family_Income_Level"              : family_income,
        "Learning_Style"                   : learning_style,
    }])

    try:
        # Encode kolom kategorikal (sama seperti saat training)
        from sklearn.preprocessing import LabelEncoder
        cat_cols = ["Gender", "Part_Time_Job", "Extracurricular_Activities",
                    "Internet_Quality", "Family_Income_Level", "Learning_Style"]
        input_enc = input_raw.copy()
        for col in cat_cols:
            le_col = LabelEncoder()
            le_col.fit(input_enc[col])
            input_enc[col] = le_col.transform(input_enc[col])

        # Impute & scale
        input_imputed = imputer.transform(input_enc[FEATURES])
        input_scaled  = scaler.transform(input_imputed)

        # Pilih model
        model_map = {"Random Forest": rf_model, "SVM": svm_model, "KNN": knn_model}
        selected_model = model_map[algoritma]

        pred_encoded = selected_model.predict(input_scaled)
        pred_label   = label_encoder.inverse_transform(pred_encoded)[0]

        result_config = {
            "High": {
                "css"  : "result-high",
                "emoji": "🌟",
                "label": "High — Performa Tinggi",
                "desc" : "Siswa ini diprediksi memiliki performa akademik yang tinggi. Pertahankan kebiasaan belajar yang baik!"
            },
            "Medium": {
                "css"  : "result-medium",
                "emoji": "📈",
                "label": "Medium — Performa Sedang",
                "desc" : "Siswa ini diprediksi memiliki performa akademik yang sedang. Masih ada ruang untuk berkembang lebih baik."
            },
            "Low": {
                "css"  : "result-low",
                "emoji": "⚠️",
                "label": "Low — Performa Rendah",
                "desc" : "Siswa ini diprediksi membutuhkan perhatian lebih. Disarankan untuk meningkatkan jam belajar dan kehadiran."
            },
        }

        cfg = result_config.get(pred_label, {
            "css": "result-medium", "emoji": "📊",
            "label": pred_label, "desc": "Prediksi berhasil dilakukan."
        })

        st.markdown(f"""
        <div class="{cfg['css']}">
            <p class="result-label">{cfg['emoji']} {cfg['label']}</p>
            <p class="result-desc">Algoritma: <strong>{algoritma}</strong> &nbsp;|&nbsp; {cfg['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        st.info("Pastikan nama kolom sesuai dengan kolom saat training. Jalankan print(X.columns.tolist()) di Colab untuk cek.")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.caption("Proyek UAS Data Science & Machine Learning · UIN Maulana Malik Ibrahim Malang")
