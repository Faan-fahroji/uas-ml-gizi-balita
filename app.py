import streamlit as st
import pandas as pd
import numpy as np

# Visualisasi
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# ML
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

# CSS ringan biar tidak kelihatan seperti tugas yang dikerjain 20 menit sebelum deadline
st.markdown(
    """
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# 2) SIDEBAR - NAVIGASI & UPLOAD
# -----------------------------
st.sidebar.title("Menu Utama")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Beranda", "📋 Ringkasan Data", "📊 Visualisasi Distribusi", "🎯 Analisis Clustering"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload file CSV/XLSX (data gizi balita)",
    type=["csv", "xlsx"],
)

st.sidebar.info(
    "Dashboard ini menganalisis kerawanan status gizi balita berdasarkan indikator "
    "**Sangat Kurang** dan **Kurang**, lalu mengelompokkan wilayah dengan K-Means."
)

# -----------------------------
# 3) LOAD DATA + CLEANING
# -----------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    if file is None:
        raise ValueError("Dataset belum di-upload.")

    name = file.name.lower()

    # Baca file
    if name.endswith(".xlsx"):
        df = pd.read_excel(file, skiprows=2, engine="openpyxl")
    else:
        df = pd.read_csv(file, skiprows=2)

    # Rapikan nama kolom
    df.columns = df.columns.astype(str).str.strip()

    # Normalisasi kemungkinan variasi nama kolom
    # (manusia suka ganti judul kolom seenaknya)
    col_map = {}
    for c in df.columns:
        c_norm = c.lower().strip()
        if c_norm in ["kabupaten/kota", "kabupaten kota", "kabupaten"]:
            col_map[c] = "Kabupaten"
        elif c_norm in ["kecamatan", "nama kecamatan"]:
            col_map[c] = "Kecamatan"
        elif c_norm in ["sangat kurang", "sangat_kurang", "gizi sangat kurang"]:
            col_map[c] = "Sangat Kurang"
        elif c_norm in ["kurang", "gizi kurang"]:
            col_map[c] = "Kurang"

    df = df.rename(columns=col_map)

    # Validasi kolom wajib
    required = ["Kabupaten", "Kecamatan", "Sangat Kurang", "Kurang"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing}\n"
            f"Kolom yang terbaca: {list(df.columns)}"
        )

    # Konversi numerik untuk fitur ML
    df["Sangat Kurang"] = pd.to_numeric(df["Sangat Kurang"], errors="coerce")
    df["Kurang"] = pd.to_numeric(df["Kurang"], errors="coerce")

    # Buang baris yang kosong/aneh
    df = df.dropna(subset=required).copy()

    # Tambah total kasus (feature engineering untuk ringkasan)
    df["Total Kasus"] = df["Sangat Kurang"] + df["Kurang"]

    return df


try:
    df = load_data(uploaded)
except Exception as e:
    if menu != "🏠 Beranda":
        st.error(f"Gagal memuat data: {e}")
        st.stop()

# -----------------------------
# 4) HALAMAN: BERANDA
# -----------------------------
if menu == "🏠 Beranda":
    st.title("🧒📊 Dashboard Analisis Status Gizi Balita 2023")
    st.markdown(
        """
        Dashboard ini membantu melihat **wilayah rawan gizi** dengan fokus pada dua indikator utama:
        1. **Sangat Kurang**
        2. **Kurang**
        
        **Fitur utama:**
        - Ringkasan statistik (total, rata-rata, wilayah tertinggi).
        - Visualisasi distribusi per wilayah.
        - **Clustering K-Means** untuk mengelompokkan wilayah yang mirip pola kasusnya.
        - Peta sebaran cluster menggunakan **PCA (reduksi dimensi)**.
        """
    )

    st.info("Upload dataset kamu di sidebar untuk mulai analisis.")

    st.markdown("### Format kolom yang disarankan")
    st.code("Kabupaten | Kecamatan | Sangat Kurang | Kurang", language="text")

# -----------------------------
# 5) HALAMAN: RINGKASAN DATA
# -----------------------------
elif menu == "📋 Ringkasan Data":
    st.header("📋 Ringkasan Statistik Data")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Kasus (SK+K)", f"{int(df['Total Kasus'].sum()):,}")
    col2.metric("Rata-rata Total per Kecamatan", f"{df['Total Kasus'].mean():.2f}")
    col3.metric("Wilayah Total Tertinggi", df.loc[df["Total Kasus"].idxmax(), "Kecamatan"])
    col4.metric("Jumlah Data (baris)", f"{len(df):,}")

    st.markdown("---")

    st.subheader("Data Lengkap")
    st.dataframe(df, use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.write(df[["Sangat Kurang", "Kurang", "Total Kasus"]].describe())

# -----------------------------
# 6) HALAMAN: VISUALISASI DISTRIBUSI
# -----------------------------
elif menu == "📊 Visualisasi Distribusi":
    st.header("📊 Visualisasi Distribusi Kasus")

    kab_list = sorted(df["Kabupaten"].unique().tolist())
    kab_pilih = st.multiselect(
        "Filter Kabupaten (opsional):",
        kab_list,
        default=kab_list[:1] if len(kab_list) > 0 else [],
    )

    view_df = df.copy()
    if kab_pilih:
        view_df = view_df[view_df["Kabupaten"].isin(kab_pilih)].copy()

    top_n = st.slider("Tampilkan Top-N Kecamatan (berdasarkan Total Kasus):", 5, 30, 10)

    top_df = (
        view_df.sort_values("Total Kasus", ascending=False)
        .head(top_n)
        .copy()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Top Kecamatan - Total Kasus")
        fig, ax = plt.subplots()
        sns.barplot(data=top_df, x="Kecamatan", y="Total Kasus", ax=ax)
        plt.xticks(rotation=60, ha="right")
        ax.set_xlabel("Kecamatan")
        ax.set_ylabel("Total Kasus (SK+K)")
        st.pyplot(fig)

    with col2:
        st.write("### Komposisi Rata-rata Indikator")
        avg_vals = view_df[["Sangat Kurang", "Kurang"]].mean()
        fig2, ax2 = plt.subplots()
        ax2.pie(avg_vals, labels=avg_vals.index, autopct="%1.1f%%", startangle=140)
        st.pyplot(fig2)

    st.markdown("---")
    st.write("### Scatter: Sangat Kurang vs Kurang")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=view_df, x="Sangat Kurang", y="Kurang", ax=ax3, s=100)
    ax3.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig3)

# -----------------------------
# 7) HALAMAN: ANALISIS CLUSTERING
# -----------------------------
elif menu == "🎯 Analisis Clustering":
    st.header("🎯 Analisis Clustering Wilayah (K-Means)")
    st.write(
        "Clustering digunakan untuk mengelompokkan kecamatan berdasarkan kemiripan pola "
        "**Sangat Kurang** dan **Kurang**."
    )

    # Fitur untuk clustering
    features = ["Sangat Kurang", "Kurang"]
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Pengaturan Clustering")
    k = st.sidebar.slider("Jumlah Cluster (K):", 2, 6, 3)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered["Cluster"] = model.fit_predict(X_scaled)

    # PCA untuk visualisasi 2D (walau fitur cuma 2, ini menjaga konsistensi gaya zaki_app)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    st.subheader("🗺️ Peta Sebaran Cluster (PCA)")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.get_cmap("tab10")

    for i in range(k):
        pts = reduced[df_clustered["Cluster"] == i]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=180,
            label=f"Cluster {i}",
            color=colors(i / 10),
            edgecolors="white",
            alpha=0.85,
        )

    # Label titik: kecamatan (bisa padat, tapi manusia suka label)
    show_labels = st.checkbox("Tampilkan label Kecamatan", value=False)
    if show_labels:
        for j, txt in enumerate(df_clustered["Kecamatan"].astype(str).tolist()):
            ax.annotate(
                txt,
                (reduced[j, 0], reduced[j, 1]),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    # Centroid
    centroid_reduced = pca.transform(model.cluster_centers_)
    ax.scatter(
        centroid_reduced[:, 0],
        centroid_reduced[:, 1],
        s=350,
        c="black",
        marker="X",
        label="Centroid",
    )

    ax.set_xlabel("Komponen Utama 1")
    ax.set_ylabel("Komponen Utama 2")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📋 Anggota Cluster")
        for i in range(k):
            with st.expander(f"Lihat Kecamatan di Cluster {i}"):
                subset = df_clustered[df_clustered["Cluster"] == i]
                st.write(", ".join(subset["Kecamatan"].astype(str).tolist()))

                st.write("**Rata-rata indikator di cluster ini:**")
                st.table(subset[features + ["Total Kasus"]].mean().to_frame().T)

    with col2:
        st.subheader("📌 Ringkasan Cluster")
        summary = (
            df_clustered.groupby("Cluster")[features + ["Total Kasus"]]
            .mean()
            .sort_values("Total Kasus", ascending=False)
        )
        st.dataframe(summary, use_container_width=True)
        st.info("Cluster dengan **rata-rata Total Kasus tertinggi** = wilayah paling rawan.")

    st.subheader("🔎 Detail Data (dengan Cluster)")
    st.dataframe(
        df_clustered[["Kabupaten", "Kecamatan", "Sangat Kurang", "Kurang", "Total Kasus", "Cluster"]],
        use_container_width=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 UAS ML - Dashboard Gizi Balita")
