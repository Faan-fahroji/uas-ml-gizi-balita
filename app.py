import re, textwrap, os

enhanced_block = textwrap.dedent("""\
    elif menu == "📊 Visualisasi Distribusi":
        st.header("📊 Visualisasi Distribusi")

        # Filter Kabupaten
        kab_list = sorted(dfc["Kabupaten"].unique().tolist())
        kab_filter = st.multiselect("Filter Kabupaten:", kab_list, default=kab_list)

        view_df = dfc[dfc["Kabupaten"].isin(kab_filter)].copy()

        # =========================
        # 1) Top Kecamatan (bar)
        # =========================
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 10 Kecamatan (Total Kasus)")
            top = view_df.sort_values("Total Kasus", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=top, x="Kecamatan", y="Total Kasus", ax=ax)
            plt.xticks(rotation=60, ha="right")
            ax.set_xlabel("Kecamatan")
            ax.set_ylabel("Total Kasus (SK+K)")
            st.pyplot(fig)

        with col2:
            st.subheader("Komposisi Rata-rata (SK vs K)")
            avg_vals = view_df[["Sangat Kurang", "Kurang"]].mean()
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.pie(avg_vals, labels=avg_vals.index, autopct="%1.1f%%", startangle=140)
            st.pyplot(fig2)

        st.markdown("---")

        # =========================
        # 2) Scatter plot SK vs K (dengan cluster)
        # =========================
        st.subheader("Scatter: Sangat Kurang vs Kurang (berdasarkan Cluster)")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=view_df, x="Sangat Kurang", y="Kurang",
            hue="Cluster", palette="viridis", s=140, ax=ax3
        )
        ax3.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig3)

        st.markdown("---")

        # =========================
        # 3) Distribusi Total Kasus (histogram)
        # =========================
        st.subheader("Distribusi Total Kasus")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.histplot(data=view_df, x="Total Kasus", kde=True, ax=ax4)
        ax4.set_xlabel("Total Kasus (SK+K)")
        ax4.set_ylabel("Frekuensi")
        ax4.grid(True, linestyle="--", alpha=0.25)
        st.pyplot(fig4)

        # =========================
        # 4) Boxplot Total Kasus per Cluster
        # =========================
        st.subheader("Perbandingan Total Kasus per Cluster (Boxplot)")
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=view_df, x="Cluster", y="Total Kasus", ax=ax5)
        ax5.set_xlabel("Cluster")
        ax5.set_ylabel("Total Kasus (SK+K)")
        ax5.grid(True, linestyle="--", alpha=0.25)
        st.pyplot(fig5)

        st.markdown("---")

        # =========================
        # 5) Jumlah Kecamatan per Cluster (countplot)
        # =========================
        st.subheader("Jumlah Kecamatan per Cluster")
        fig6, ax6 = plt.subplots(figsize=(10, 5))
        sns.countplot(data=view_df, x="Cluster", ax=ax6)
        ax6.set_xlabel("Cluster")
        ax6.set_ylabel("Jumlah Kecamatan")
        ax6.grid(True, linestyle="--", alpha=0.25)
        st.pyplot(fig6)

        # =========================
        # 6) Korelasi fitur numerik (heatmap)
        # =========================
        st.subheader("Korelasi Indikator (Heatmap)")
        num_cols = ["Sangat Kurang", "Kurang", "Total Kasus"]
        corr = view_df[num_cols].corr(numeric_only=True)

        fig7, ax7 = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax7)
        st.pyplot(fig7)

    # =============================
    # HALAMAN: ANALISIS CLUSTERING
    # =============================
    elif menu == "🎯 Analisis Clustering":
""")

app2 = re.sub(
    r"elif menu == \"📊 Visualisasi Distribusi\":.*?elif menu == \"🎯 Analisis Clustering\":",
    enhanced_block,
    app,
    flags=re.S
)

with open(app_path,'w',encoding='utf-8') as f:
    f.write(app2)

# re-zip
zip_out="/mnt/data/gizi_balita_streamlit_project_v2.zip"
if os.path.exists(zip_out):
    os.remove(zip_out)
with zipfile.ZipFile(zip_out,'w',zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(work):
        for file in files:
            full=os.path.join(root,file)
            rel=os.path.relpath(full, work)
            z.write(full, arcname=rel)

zip_out

