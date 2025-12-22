import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------
# 1) KONFIGURASI HALAMAN
# -----------------------------
st.set_page_config(
    page_title="Analisis Status Gizi Balita 2023",
    page_icon="🧒📊",
    layout="wide",
)

st.st.markdown("""
<style>
/* Background aplikasi (biar konsisten) */
.main { 
    background-color: #0e1117; /* cocok buat dark */
}

/* Kartu metric */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    padding: 16px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.10) !important;
}

/* Label metric (judul kecil) */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] p {
    color: #555555 !important;
    font-weight: 600 !important;
}

/* Angka metric (nilai besar) */
div[data-testid="stMetric"] div {
    color: #111111 !important;
    font-weight: 800 !important;
}

/* Kadang Streamlit bungkus angka di span */
div[data-testid="stMetric"] span {
    color: #111111 !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

)

# -----------------------------
# 2) PATH DATASET (LANGSUNG PANGGIL)
# -----------------------------
DATA_FILE = "Hasil_Clustering.csv"   # <-- dataset kamu

# -----------------------------
# 3) LOAD DATA
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # rapikan nama kolom
    df.columns = df.columns.astype(str).str.strip()

    # Kalau dataset kamu sudah punya kolom cluster, kita pakai itu.
    # Kalau belum ada, nanti dihitung ulang di menu clustering.
    return df


try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(
        f"Gagal memuat dataset '{DATA_FILE}'. Pastikan file ada satu folder dengan app.py.\n\nDetail: {e}"
    )
    st.stop()

# -----------------------------
# 4) SIDEBAR NAVIGASI
# -----------------------------
st.sidebar.title("Menu Utama")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Beranda", "📋 Ringkasan Data", "📊 Visualisasi Distribusi", "🎯 Analisis Clustering"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"Dataset aktif: **{DATA_FILE}**\n\n"
    "Dashboard ini menganalisis wilayah rawan gizi dan clustering K-Means."
)

# -----------------------------
# 5) BERANDA
# -----------------------------
if menu == "🏠 Beranda":
    st.title("🧒📊 Dashboard Analisis Status Gizi Balita 2023")
    st.write("Dataset dipanggil langsung dari file CSV di project.")
    st.success(f"Berhasil memuat data: {len(df):,} baris, {len(df.columns):,} kolom.")
    st.subheader("Preview Data")
    st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# 6) RINGKASAN DATA
# -----------------------------
elif menu == "📋 Ringkasan Data":
    st.header("📋 Ringkasan Data")

    # Coba deteksi kolom numerik utama (kamu bisa sesuaikan kalau beda nama)
    # Prioritas: "Sangat Kurang" dan "Kurang" (umum di dataset gizi)
    cols = df.columns.tolist()

    kandidat_sk = [c for c in cols if c.lower() in ["sangat kurang", "sangat_kurang", "sk"]]
    kandidat_k = [c for c in cols if c.lower() in ["kurang", "k"]]

    sk_col = kandidat_sk[0] if kandidat_sk else None
    k_col = kandidat_k[0] if kandidat_k else None

    if sk_col and k_col:
        # bersihkan numeric
        dfx = df.copy()
        dfx[sk_col] = pd.to_numeric(dfx[sk_col], errors="coerce")
        dfx[k_col] = pd.to_numeric(dfx[k_col], errors="coerce")
        dfx["Total Kasus"] = dfx[sk_col] + dfx[k_col]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Kasus (SK+K)", f"{int(dfx['Total Kasus'].sum()):,}")
        col2.metric("Rata-rata Total", f"{dfx['Total Kasus'].mean():.2f}")
        col3.metric("Max SK", f"{int(dfx[sk_col].max()):,}")
        col4.metric("Jumlah Baris", f"{len(dfx):,}")

        st.markdown("---")
        st.subheader("Data Lengkap")
        st.dataframe(dfx, use_container_width=True)
        st.subheader("Statistik Deskriptif")
        st.write(dfx[[sk_col, k_col, "Total Kasus"]].describe())

    else:
        st.warning(
            "Kolom 'Sangat Kurang' / 'Kurang' tidak terdeteksi otomatis.\n"
            "Tampilkan saja data mentah dulu."
        )
        st.dataframe(df, use_container_width=True)

# -----------------------------
# 7) VISUALISASI DISTRIBUSI
# -----------------------------
elif menu == "📊 Visualisasi Distribusi":
    st.header("📊 Visualisasi Distribusi")

    # cari kolom numeric untuk plot cepat
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Kolom numerik kurang dari 2, tidak bisa bikin scatter/bar chart.")
        st.dataframe(df.head(50), use_container_width=True)
        st.stop()

    x_col = st.selectbox("Pilih sumbu X:", numeric_cols, index=0)
    y_col = st.selectbox("Pilih sumbu Y:", numeric_cols, index=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, s=120)
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top 10 berdasarkan Y")
    top = df.sort_values(y_col, ascending=False).head(10)
    st.dataframe(top, use_container_width=True)

# -----------------------------
# 8) CLUSTERING (K-MEANS + PCA)
# -----------------------------
elif menu == "🎯 Analisis Clustering":
    st.header("🎯 Analisis Clustering Wilayah (K-Means)")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Kolom numerik kurang dari 2 untuk clustering.")
        st.stop()

    st.write("Pilih fitur numerik untuk clustering:")
    features = st.multiselect(
        "Fitur:",
        numeric_cols,
        default=numeric_cols[:2],
    )

    if len(features) < 2:
        st.warning("Minimal pilih 2 fitur.")
        st.stop()

    k = st.sidebar.slider("Jumlah Cluster (K):", 2, 6, 3)

    X = df[features].copy()
    X = X.apply(pd.to_numeric, errors="coerce").dropna()
    df_used = df.loc[X.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_used["Cluster"] = model.fit_predict(X_scaled)

    # PCA untuk visualisasi 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    st.subheader("🗺️ Peta Cluster (PCA 2D)")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.get_cmap("tab10")

    for i in range(k):
        pts = reduced[df_used["Cluster"] == i]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=160,
            label=f"Cluster {i}",
            color=colors(i / 10),
            edgecolors="white",
            alpha=0.85,
        )

    centroids_2d = pca.transform(model.cluster_centers_)
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="black", s=320, marker="X", label="Centroid")

    ax.set_xlabel("Komponen Utama 1")
    ax.set_ylabel("Komponen Utama 2")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📋 Ringkasan Cluster (rata-rata fitur)")
    st.dataframe(df_used.groupby("Cluster")[features].mean(), use_container_width=True)

    st.subheader("🔎 Data dengan Cluster")
    st.dataframe(df_used, use_container_width=True)

