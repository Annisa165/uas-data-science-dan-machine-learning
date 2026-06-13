import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Performa Akademik",
    page_icon="🎓",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

# Fitur asli yang dipakai model saat training
FEATURES = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Attendance_Percentage",
    "Extracurricular_Activities",
    "Screen_Time_Per_Day",
    "Nutrition_Quality",
    "Stress_Level",
    "Digital_Tools_Usage",
    "Assignments_On_Time",
]

# Mapping manual agar input kategori berubah jadi angka
EXTRACURRICULAR_MAP = {"No": 0, "Yes": 1}
NUTRITION_MAP = {"Poor": 0, "Average": 1, "Good": 2}
DIGITAL_TOOLS_MAP = {"Low": 0, "Medium": 1, "High": 2}

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
    .hero h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
    }
    .hero p {
        font-size: 0.95rem;
        opacity: 0.85;
        margin: 0;
    }

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
    .result-average {
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
    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .result-desc {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0;
    }

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
    filenames = {
        "Random Forest": "model_rf.pkl",
        "SVM": "model_svm.pkl",
        "KNN": "model_knn.pkl",
        "scaler": "scaler.pkl",
        "imputer": "imputer.pkl",
        "label_encoder": "label_encoder.pkl",
    }

    loaded = {}

    for key, filename in filenames.items():
        file_path = BASE_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {filename}")

        with open(file_path, "rb") as f:
            loaded[key] = pickle.load(f)

    return loaded


try:
    artifacts = load_artifacts()
    model_loaded = True
    load_error = None
except Exception as e:
    artifacts = {}
    model_loaded = False
    load_error = e

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎓 Prediksi Performa Akademik Siswa</h1>
    <p>Masukkan data kebiasaan belajar, kehadiran, gaya hidup, dan aktivitas siswa untuk memprediksi performa akademik.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Model gagal dimuat: {load_error}")
    st.info(
        "Pastikan semua file .pkl berada di folder yang sama dengan app.py: "
        "model_rf.pkl, model_svm.pkl, model_knn.pkl, scaler.pkl, imputer.pkl, label_encoder.pkl"
    )
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
# FORM INPUT SESUAI FITUR TRAINING
# ─────────────────────────────────────────
st.markdown('<div class="card"><h3>📋 Data Siswa</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Jam belajar per hari",
        min_value=0.0,
        max_value=12.0,
        value=4.0,
        step=0.5
    )

    sleep_hours = st.slider(
        "Jam tidur per hari",
        min_value=0.0,
        max_value=12.0,
        value=7.0,
        step=0.5
    )

    attendance = st.slider(
        "Tingkat kehadiran (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    screen_time = st.slider(
        "Screen time per hari (jam)",
        min_value=0.0,
        max_value=16.0,
        value=5.0,
        step=0.5
    )

    assignments_on_time = st.slider(
        "Tugas selesai tepat waktu",
        min_value=0,
        max_value=10,
        value=5
    )

with col2:
    extracurricular = st.selectbox(
        "Aktif ekstrakurikuler?",
        ["No", "Yes"]
    )

    nutrition_quality = st.selectbox(
        "Kualitas nutrisi",
        ["Poor", "Average", "Good"]
    )

    stress_level = st.slider(
        "Tingkat stres (1–10)",
        min_value=1,
        max_value=10,
        value=6
    )

    digital_tools_usage = st.selectbox(
        "Penggunaan alat digital belajar",
        ["Low", "Medium", "High"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang"):

    input_data = pd.DataFrame([{
        "Study_Hours_Per_Day": study_hours,
        "Sleep_Hours_Per_Day": sleep_hours,
        "Attendance_Percentage": attendance,
        "Extracurricular_Activities": EXTRACURRICULAR_MAP[extracurricular],
        "Screen_Time_Per_Day": screen_time,
        "Nutrition_Quality": NUTRITION_MAP[nutrition_quality],
        "Stress_Level": stress_level,
        "Digital_Tools_Usage": DIGITAL_TOOLS_MAP[digital_tools_usage],
        "Assignments_On_Time": assignments_on_time,
    }], columns=FEATURES)

    try:
        imputer = artifacts["imputer"]
        scaler = artifacts["scaler"]
        label_encoder = artifacts["label_encoder"]
        selected_model = artifacts[algoritma]

        input_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_imputed)

        pred_encoded = selected_model.predict(input_scaled)
        pred_label = label_encoder.inverse_transform(pred_encoded)[0]

        result_config = {
            "High": {
                "css": "result-high",
                "emoji": "🌟",
                "label": "High — Performa Tinggi",
                "desc": "Siswa ini diprediksi memiliki performa akademik yang tinggi. Pertahankan kebiasaan belajar yang baik."
            },
            "Average": {
                "css": "result-average",
                "emoji": "📈",
                "label": "Average — Performa Rata-rata",
                "desc": "Siswa ini diprediksi memiliki performa akademik rata-rata. Masih ada ruang untuk ditingkatkan."
            },
            "Low": {
                "css": "result-low",
                "emoji": "⚠️",
                "label": "Low — Performa Rendah",
                "desc": "Siswa ini diprediksi membutuhkan perhatian lebih, terutama pada kebiasaan belajar, kehadiran, dan penyelesaian tugas."
            },
        }

        cfg = result_config.get(str(pred_label), {
            "css": "result-average",
            "emoji": "📊",
            "label": str(pred_label),
            "desc": "Prediksi berhasil dilakukan."
        })

        st.markdown(f"""
        <div class="{cfg['css']}">
            <p class="result-label">{cfg['emoji']} {cfg['label']}</p>
            <p class="result-desc">
                Algoritma: <strong>{algoritma}</strong> &nbsp;|&nbsp; {cfg['desc']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Lihat data input yang dipakai model"):
            st.dataframe(input_data)

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        st.info(
            "Cek kembali apakah file model, scaler, imputer, dan label_encoder "
            "dibuat dari dataset serta urutan fitur yang sama."
        )

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.caption("Proyek UAS Data Science & Machine Learning · UIN Maulana Malik Ibrahim Malang")
