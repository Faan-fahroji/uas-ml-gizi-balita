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
# VISUALISASI DISTRIBUSI (VERSI LEBIH KOMPLEKS)
# =============================
elif menu == "📊 Visualisasi Distribusi":
    st.header("📊 Visualisasi Distribusi (Lengkap)")

    # --- Pastikan kolom utama ada ---
    required = ["Sangat Kurang", "Kurang"]
    for c in required:
        if c not in df.columns:
            st.error(f"Kolom '{c}' tidak ditemukan di dataset.")
            st.stop()

    # Konversi numeric (biar aman kalau ada string)
    view_df = df.copy()
    view_df["Sangat Kurang"] = pd.to_numeric(view_df["Sangat Kurang"], errors="coerce")
    view_df["Kurang"] = pd.to_numeric(view_df["Kurang"], errors="coerce")
    view_df = view_df.dropna(subset=["Sangat Kurang", "Kurang"]).copy()
    view_df["Total Kasus"] = view_df["Sangat Kurang"] + view_df["Kurang"]

    # --- Filter Kabupaten (kalau ada) ---
    if "Kabupaten" in view_df.columns:
        kab_list = sorted(view_df["Kabupaten"].astype(str).unique().tolist())
        kab_filter = st.multiselect("Filter Kabupaten:", kab_list, default=kab_list)
        view_df = view_df[view_df["Kabupaten"].astype(str).isin(kab_filter)].copy()

    # --- Filter Cluster (kalau ada) ---
    if "Cluster" in view_df.columns:
        cluster_list = sorted(view_df["Cluster"].dropna().unique().tolist())
        cluster_filter = st.multiselect("Filter Cluster:", cluster_list, default=cluster_list)
        view_df = view_df[view_df["Cluster"].isin(cluster_filter)].copy()

    st.caption(f"Data setelah filter: {len(view_df):,} baris")

    # ======================================================
    # TAB BIAR RAPI (dan kelihatan niat)
    # ======================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Ranking Wilayah",
        "📈 Distribusi & Outlier",
        "🧭 Hubungan Variabel",
        "🧩 Ringkasan Cluster"
    ])

    # ======================================================
    # TAB 1: Ranking Wilayah (Top N)
    # ======================================================
    with tab1:
        st.subheader("📌 Top Kecamatan Berdasarkan Total Kasus")
        top_n = st.slider("Tampilkan Top-N:", 5, 30, 10)

        if "Kecamatan" in view_df.columns:
            top_df = view_df.sort_values("Total Kasus", ascending=False).head(top_n)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=top_df, x="Kecamatan", y="Total Kasus", ax=ax)
            plt.xticks(rotation=60, ha="right")
            ax.set_xlabel("Kecamatan")
            ax.set_ylabel("Total Kasus (SK+K)")
            ax.grid(True, linestyle="--", alpha=0.25)
            st.pyplot(fig)

            st.dataframe(top_df[["Kabupaten","Kecamatan","Sangat Kurang","Kurang","Total Kasus"]]
                         if "Kabupaten" in top_df.columns else
                         top_df[["Kecamatan","Sangat Kurang","Kurang","Total Kasus"]],
                         use_container_width=True)
        else:
            st.warning("Kolom 'Kecamatan' tidak ditemukan, jadi ranking kecamatan tidak bisa dibuat.")

        st.markdown("---")
        st.subheader("📌 Komposisi Rata-rata Indikator")
        avg_vals = view_df[["Sangat Kurang", "Kurang"]].mean()
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(avg_vals, labels=avg_vals.index, autopct="%1.1f%%", startangle=140)
        st.pyplot(fig2)

    # ======================================================
    # TAB 2: Distribusi & Outlier
    # ======================================================
    with tab2:
        st.subheader("📈 Distribusi Total Kasus")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.histplot(data=view_df, x="Total Kasus", kde=True, ax=ax3)
        ax3.set_xlabel("Total Kasus (SK+K)")
        ax3.set_ylabel("Frekuensi")
        ax3.grid(True, linestyle="--", alpha=0.25)
        st.pyplot(fig3)

        st.markdown("---")
        st.subheader("📦 Boxplot Total Kasus (Outlier)")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=view_df, y="Total Kasus", ax=ax4)
        ax4.set_ylabel("Total Kasus (SK+K)")
        ax4.grid(True, linestyle="--", alpha=0.25)
        st.pyplot(fig4)

        st.markdown("---")
        st.subheader("📦 Boxplot Sangat Kurang vs Kurang")
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        plot_df = view_df[["Sangat Kurang", "Kurang"]].melt(var_name="Indikator", value_name="Jumlah")
        sns.boxplot(data=plot_df, x="Indikator", y="Jumlah", ax=ax5)
        ax5.grid(True, linestyle="--", alpha=0.25)
        st.pyplot(fig5)

    # ======================================================
    # TAB 3: Hubungan Variabel
    # ======================================================
    with tab3:
        st.subheader("🧭 Scatter: Sangat Kurang vs Kurang")
        fig6, ax6 = plt.subplots(figsize=(10, 6))

        if "Cluster" in view_df.columns:
            sns.scatterplot(
                data=view_df,
                x="Sangat Kurang",
                y="Kurang",
                hue="Cluster",
                palette="viridis",
                s=140,
                ax=ax6
            )
        else:
            sns.scatterplot(
                data=view_df,
                x="Sangat Kurang",
                y="Kurang",
                s=140,
                ax=ax6
            )

        ax6.grid(True, linestyle="--", alpha=0.35)
        st.pyplot(fig6)

        st.markdown("---")
        st.subheader("🔥 Heatmap Korelasi")
        corr_cols = ["Sangat Kurang", "Kurang", "Total Kasus"]
        corr = view_df[corr_cols].corr(numeric_only=True)

        fig7, ax7 = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax7)
        st.pyplot(fig7)

    # ======================================================
    # TAB 4: Ringkasan Cluster (kalau ada cluster)
    # ======================================================
    with tab4:
        if "Cluster" not in view_df.columns:
            st.info("Kolom 'Cluster' tidak ada di dataset/hasil filter. Bagian ringkasan cluster tidak ditampilkan.")
        else:
            st.subheader("🧩 Jumlah Kecamatan per Cluster")
            fig8, ax8 = plt.subplots(figsize=(10, 5))
            sns.countplot(data=view_df, x="Cluster", ax=ax8)
            ax8.set_xlabel("Cluster")
            ax8.set_ylabel("Jumlah Data")
            ax8.grid(True, linestyle="--", alpha=0.25)
            st.pyplot(fig8)

            st.markdown("---")
            st.subheader("🧩 Rata-rata Indikator per Cluster")
            st.dataframe(
                view_df.groupby("Cluster")[["Sangat Kurang","Kurang","Total Kasus"]]
                .mean()
                .sort_values("Total Kasus", ascending=False),
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("📌 Interpretasi cepat")
            st.info("Cluster dengan rata-rata **Total Kasus tertinggi** dapat dianggap kelompok wilayah paling rawan.")
