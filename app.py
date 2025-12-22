import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Analisis Status Gizi Balita 2023",
    page_icon="🧒📊",
    layout="wide",
)

# =============================
# CSS FINAL (FIX METRIC DARK MODE)
# =============================
st.markdown("""
<style>
.main { background-color: #0e1117; }

/* Kartu metric */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    padding: 16px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12) !important;
}

/* Label metric */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] p {
    color: #555555 !important;
    font-weight: 600 !important;
}

/* Nilai metric */
div[data-testid="stMetric"] div,
div[data-testid="stMetric"] span {
    color: #111111 !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# PATH DATASET
# =============================
DATA_FILE = "Hasil_Clustering.csv"

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df

try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(f"Gagal memuat dataset '{DATA_FILE}'. Pastikan file satu folder dengan app.py.\n\n{e}")
    st.stop()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("Menu Utama")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Beranda", "📋 Ringkasan Data", "📊 Visualisasi Distribusi", "🎯 Analisis Clustering"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"Dataset aktif: **{DATA_FILE}**")

# =============================
# BERANDA
# =============================
if menu == "🏠 Beranda":
    st.title("🧒📊 Dashboard Analisis Status Gizi Balita 2023")
    st.write(
        "Dashboard ini digunakan untuk menganalisis wilayah karawang status gizi balita "
        "berdasarkan indikator **Sangat Kurang (SK)** dan **Kurang (K)**."
    )

    st.success(f"Data berhasil dimuat: {len(df):,} baris, {len(df.columns):,} kolom")
    st.subheader("Preview Data")
    st.dataframe(df.head(20), use_container_width=True)

# =============================
# RINGKASAN DATA
# =============================
elif menu == "📋 Ringkasan Data":
    st.header("📋 Ringkasan Data")

    # Deteksi kolom penting
    cols = df.columns.str.lower().tolist()
    sk_col = next((c for c in df.columns if c.lower() in ["sangat kurang", "sangat_kurang"]), None)
    k_col  = next((c for c in df.columns if c.lower() in ["kurang"]), None)

    if sk_col and k_col:
        dfx = df.copy()
        dfx[sk_col] = pd.to_numeric(dfx[sk_col], errors="coerce")
        dfx[k_col]  = pd.to_numeric(dfx[k_col], errors="coerce")
        dfx["Total Kasus"] = dfx[sk_col] + dfx[k_col]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Kasus (SK + K)", f"{int(dfx['Total Kasus'].sum()):,}")
        col2.metric("Rata-rata Total", f"{dfx['Total Kasus'].mean():.2f}")
        col3.metric("Nilai SK Tertinggi", f"{int(dfx[sk_col].max()):,}")
        col4.metric("Jumlah Data", f"{len(dfx):,}")

        st.markdown("---")
        st.subheader("Data Lengkap")
        st.dataframe(dfx, use_container_width=True)

        st.subheader("Statistik Deskriptif")
        st.write(dfx[[sk_col, k_col, "Total Kasus"]].describe())

    else:
        st.warning("Kolom 'Sangat Kurang' dan 'Kurang' tidak terdeteksi.")
        st.dataframe(df, use_container_width=True)

# =============================
# VISUALISASI DISTRIBUSI
# =============================
elif menu == "📊 Visualisasi Distribusi":
    st.header("📊 Visualisasi Distribusi")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Kolom numerik tidak cukup untuk visualisasi.")
        st.stop()

    x_col = st.selectbox("Sumbu X", numeric_cols, index=0)
    y_col = st.selectbox("Sumbu Y", numeric_cols, index=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, s=130, ax=ax)
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

# =============================
# ANALISIS CLUSTERING
# =============================
elif menu == "🎯 Analisis Clustering":
    st.header("🎯 Analisis Clustering (K-Means)")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = st.multiselect(
        "Pilih fitur numerik:",
        numeric_cols,
        default=numeric_cols[:2]
    )

    if len(features) < 2:
        st.warning("Minimal pilih 2 fitur.")
        st.stop()

    k = st.sidebar.slider("Jumlah Cluster (K)", 2, 6, 3)

    X = df[features].apply(pd.to_numeric, errors="coerce").dropna()
    df_used = df.loc[X.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_used["Cluster"] = model.fit_predict(X_scaled)

    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    st.subheader("🗺️ Visualisasi Cluster (PCA)")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.get_cmap("tab10")

    for i in range(k):
        pts = reduced[df_used["Cluster"] == i]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=160,
            label=f"Cluster {i}",
            color=colors(i / 10),
            edgecolors="white"
        )

    centroids = pca.transform(model.cluster_centers_)
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c="black", s=350, marker="X", label="Centroid")

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    st.pyplot(fig)

    st.subheader("📋 Rata-rata Tiap Cluster")
    st.dataframe(df_used.groupby("Cluster")[features].mean(), use_container_width=True)

    st.subheader("🔍 Data dengan Cluster")
    st.dataframe(df_used, use_container_width=True)

# =============================
# FOOTER
# =============================
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 UAS Machine Learning – Status Gizi Balita")
