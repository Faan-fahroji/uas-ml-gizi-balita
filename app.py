import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Judul Utama Aplikasi
st.set_page_config(page_title="UAS ML - Gizi Balita", layout="wide")
st.title("📊 Aplikasi Clustering Status Gizi Balita 2023")
st.markdown("Analisis Wilayah Kerawanan Gizi di Kabupaten Purwakarta & Karawang")

# --- POINT PENTING: PATH DATA ---
# Pastikan nama file ini SAMA PERSIS dengan file yang kamu upload ke GitHub
file_name = 'data_gizi.csv.xlsx' 

@st.cache_data
def load_data():
    try:
        # Mencoba membaca format Excel/CSV
        df = pd.read_excel(file_name, skiprows=2)
    except:
        df = pd.read_csv(file_name, skiprows=2)
    
    # Cleaning Nama Kolom
    df.columns = df.columns.str.strip()
    df['Sangat Kurang'] = pd.to_numeric(df['Sangat Kurang'], errors='coerce')
    df['Kurang'] = pd.to_numeric(df['Kurang'], errors='coerce')
    return df.dropna(subset=['Sangat Kurang', 'Kurang'])

# Load Dataset
df = load_data()

# Sidebar Pengaturan
st.sidebar.header("⚙️ Konfigurasi")
k_value = st.sidebar.slider("Pilih Jumlah Cluster (K)", min_value=2, max_value=5, value=3)

# Proses Machine Learning (K-Means)
X = df[['Sangat Kurang', 'Kurang']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = model.fit_predict(X_scaled)

# Layout Kolom
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Visualisasi Sebaran Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Sangat Kurang', y='Kurang', hue='Cluster', palette='viridis', s=150, ax=ax)
    
    # Titik Pusat
    centroids = scaler.inverse_transform(model.cluster_centers_)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, marker='X', label='Titik Pusat')
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

with col2:
    st.subheader("📋 Rata-rata per Cluster")
    st.write(df.groupby('Cluster')[['Sangat Kurang', 'Kurang']].mean())
    st.info("Cluster dengan angka tertinggi menunjukkan wilayah yang paling rawan.")

# Tampilkan Tabel
st.subheader("🔍 Detail Data Kecamatan")
st.dataframe(df[['Kabupaten', 'Kecamatan', 'Sangat Kurang', 'Kurang', 'Cluster']])
