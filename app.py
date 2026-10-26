from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.pkl"
EVAL_PATH = BASE_DIR / "models" / "evaluation.json"
DATA_PATH = BASE_DIR / "dataset_ekspedisi.csv"

st.set_page_config(page_title="Klasifikasi Keterlambatan Pengiriman", page_icon="📦", layout="wide")
st.title("📦 Klasifikasi Keterlambatan Pengiriman — Regresi Logistik")

if not MODEL_PATH.exists() or not EVAL_PATH.exists():
    st.error("Model belum tersedia. Jalankan `python train_model.py` terlebih dahulu di terminal.")
    st.stop()

@st.cache_resource
def load_resources():
    model = joblib.load(str(MODEL_PATH))
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        evaluation = json.load(f)
    data = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Status_Keterlambatan"])
    return model, evaluation, data, train_df, test_df

model, evaluation, df_data, df_train, df_test = load_resources()
B0, B1, B2, B3 = evaluation["parameters"]["beta_0"], evaluation["parameters"]["beta_1"], evaluation["parameters"]["beta_2"], evaluation["parameters"]["beta_3"]
LAYANAN_MAP = evaluation["layanan_encoding"]
DEFAULT_TH = evaluation.get("threshold", 0.25)

tab_prediksi, tab_evaluasi, tab_data = st.tabs(["🔍 Prediksi Pengiriman", "📊 Evaluasi Model", "📁 Data Split & Dataset"])

with tab_prediksi:
    st.subheader("Simulasi & Prediksi Keterlambatan")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        jarak = st.number_input("Jarak Pengiriman (X₁ KM)", min_value=0.0, max_value=100000.0, value=200.0, step=10.0)
    with c2:
        transit = st.number_input("Jumlah Titik Transit (X₂)", min_value=0, max_value=20, value=1, step=1)
    with c3:
        layanan = st.selectbox("Jenis Layanan (X₃)", list(LAYANAN_MAP.keys()), index=3)
    with c4:
        threshold = st.slider("Ambang Batas (Threshold)", 0.10, 0.50, float(DEFAULT_TH), 0.01)

    layanan_code = LAYANAN_MAP[layanan]
    z = B0 + (B1 * jarak) + (B2 * transit) + (B3 * layanan_code)
    prob_terlambat = 1.0 / (1.0 + np.exp(-z))
    prob_tepat = 1.0 - prob_terlambat
    prediksi = 1 if prob_terlambat >= threshold else 0

    st.divider()
    res1, res2 = st.columns([1, 1])
    with res1:
        if prediksi == 1:
            st.error("🚨 **Hasil Prediksi: Terlambat (Y = 1)**")
        else:
            st.success("✅ **Hasil Prediksi: Tepat Waktu (Y = 0)**")
        st.write(f"• **P(Terlambat)**: `{prob_terlambat:.4f}` ({prob_terlambat * 100:.2f}%)")
        st.write(f"• **P(Tepat Waktu)**: `{prob_tepat:.4f}` ({prob_tepat * 100:.2f}%)")
        st.write(f"• **Ambang Batas Keputusan**: `{threshold:.2f}` ({threshold * 100:.0f}%)")

        fig, ax = plt.subplots(figsize=(6, 2))
        colors = ['#1f77b4', '#d62728'] if prediksi == 1 else ['#2ca02c', '#ff7f0e']
        bars = ax.barh(['Tepat Waktu', 'Terlambat'], [prob_tepat, prob_terlambat], color=colors, height=0.5)
        ax.set_xlim(0, 1)
        for bar in bars:
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2%}", va='center', fontweight='bold')
        ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
        ax.legend(loc='lower right', fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    with res2:
        st.markdown("**Perhitungan Matematis Logit & Sigmoid:**")
        st.latex(r"z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3")
        st.code(f"z = {B0} + ({B1}*{jarak:.2f}) + ({B2}*{transit}) + ({B3}*{layanan_code}) = {z:.4f}", language="text")
        st.latex(r"P(Y=1 \mid X) = \frac{1}{1 + e^{-z}}")
        st.code(f"P = 1 / (1 + e^(-({z:.4f}))) = {prob_terlambat:.4f}", language="text")
        st.caption(f"Keputusan: Karena P ({prob_terlambat:.4f}) {'≥' if prediksi==1 else '<'} {threshold:.2f}, maka paket diprediksi **{'Terlambat' if prediksi==1 else 'Tepat Waktu'}**.")

with tab_evaluasi:
    st.subheader(f"Evaluasi Model (Ambang Batas Keputusan = {DEFAULT_TH:.2f})")
    pilihan = st.radio("Pilih Data Evaluasi:", ["Seluruh Dataset (500 Data)", "Data Uji (100 Data)"], horizontal=True)
    eval_data = evaluation["evaluasi_seluruh_data"] if "500" in pilihan else evaluation["evaluasi_data_uji"]
    cm = eval_data["confusion_matrix"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", f"{eval_data['accuracy']:.2%}")
    m2.metric("Presisi", f"{eval_data['precision']:.2%}")
    m3.metric("Recall (Sensitivitas)", f"{eval_data['recall']:.2%}")
    m4.metric("F1-Score", f"{eval_data['f1_score']:.2%}")

    st.markdown("---")
    c_cm, c_info = st.columns([1, 1.2])
    with c_cm:
        st.markdown("**Confusion Matrix:**")
        cm_df = pd.DataFrame([
            {"Aktual": "Aktual Terlambat (1)", "Prediksi Terlambat (1)": f"TP = {cm['TP']}", "Prediksi Tepat Waktu (0)": f"FN = {cm['FN']}"},
            {"Aktual": "Aktual Tepat Waktu (0)", "Prediksi Terlambat (1)": f"FP = {cm['FP']}", "Prediksi Tepat Waktu (0)": f"TN = {cm['TN']}"}
        ])
        st.dataframe(cm_df, width="stretch", hide_index=True)
    with c_info:
        st.markdown("**Catatan Penyesuaian Threshold (0.25):**")
        st.info(f"Model menjaring **{cm['TP']} paket terlambat** (Recall {eval_data['recall']:.2%}) pada threshold {DEFAULT_TH:.2f}.")

    if "Data Uji" in pilihan:
        st.markdown("#### 🧪 Tabel 100 Data Uji (Hasil Split 20%):")
        df_test_eval = df_test.copy()
        z_t = B0 + (B1 * df_test_eval["Jarak_Pengiriman_KM"]) + (B2 * df_test_eval["Jumlah_Transit"]) + (B3 * df_test_eval["Jenis_Layanan"].map(LAYANAN_MAP))
        df_test_eval["Prob_Terlambat"] = (1.0 / (1.0 + np.exp(-z_t))).round(4)
        df_test_eval["Prediksi"] = np.where(df_test_eval["Prob_Terlambat"] >= DEFAULT_TH, "Terlambat", "Tepat Waktu")
        df_test_eval["Hasil"] = np.where(df_test_eval["Prediksi"] == df_test_eval["Status_Keterlambatan"], "Benar ✅", "Salah ❌")
        st.dataframe(df_test_eval, width="stretch")

with tab_data:
    st.subheader("Data Pengiriman & Hasil Pembagian (Split 80:20)")
    pilih_split = st.radio("Pilih Tampilan Data:", ["🌐 Seluruh Data (500 Baris)", "🚂 Data Latih / Train (80% = 400 Baris)", "🧪 Data Uji / Test (20% = 100 Baris)"], horizontal=True)
    tampil_df = df_train if "Train" in pilih_split else (df_test if "Test" in pilih_split else df_data)
    
    c_d1, c_d2 = st.columns(2)
    c_d1.metric("Total Baris Data", len(tampil_df))
    c_d2.write(f"**Distribusi Target ({len(tampil_df)} Baris):**")
    c_d2.dataframe(tampil_df["Status_Keterlambatan"].value_counts(), width="stretch")

    st.dataframe(tampil_df, width="stretch")
    st.download_button("📥 Unduh CSV", data=tampil_df.to_csv(index=False).encode("utf-8"), file_name=f"{pilih_split[:15]}.csv", mime="text/csv")
