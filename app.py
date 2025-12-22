import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="UAS ML - Gizi Balita", layout="wide")
st.title("📊 Aplikasi Clustering Status Gizi Balita 2023")
st.markdown("Analisis Wilayah Kerawanan Gizi di Kabupaten Purwakarta & Karawang")

st.sidebar.header("📁 Upload Dataset (CSV/XLSX)")
uploaded = st.sidebar.file_uploader("Upload file data", type=["csv", "xlsx"])

@st.cache_data
def load_data(file) -> pd.DataFrame:
    # Baca sesuai tipe file
    if file.name.endswith(".xlsx"):
        df = pd.read_excel(file, skiprows=2, engine="openpyxl")
    else:
        df = pd.read_csv(file, skiprows=2)

    # Rapikan nama kolom
    df.columns = df.columns.astype(str).str.strip()

    # Validasi kolom wajib
    required_cols = ["Sangat Kurang", "Kurang", "Kabupaten", "Kecamatan"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}. Kolom yang ada: {list(df.columns)}")

    # Konversi numerik
    df["Sangat Kurang"] = pd.to_numeric(df["Sangat Kurang"], errors="coerce")
    df["Kurang"] = pd.to_numeric(df["Kurang"], errors="coerce")

    # Buang baris yang rusak
    df = df.dropna(subset=["Sangat Kurang", "Kurang", "Kabupaten", "Kecamatan"])
    return df

if uploaded is None:
    st.warning("Silakan upload dataset dulu lewat sidebar.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

st.sidebar.header("⚙️ Konfigurasi")
k_value = st.sidebar.slider("Pilih Jumlah Cluster (K)", 2, 5, 3)

# Machine Learning
