import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="World Development Clustering", layout="wide")

st.title("🌍 World Development Clustering App (2D + 3D)")

# File uploader
uploaded_file = st.file_uploader(r"https://raw.githubusercontent.com/username/repo/main/World_development_mesurement.csv")
if uploaded_file:
    # Load file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    all_cols = df.columns.tolist()
    selected_features = st.multiselect("Select features for clustering", all_cols)

    if selected_features:
        # Prepare data
        data = df[selected_features].dropna()
        data_encoded = pd.get_dummies(data)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_encoded)

        # Clustering
        k = st.slider("Select number of clusters (K)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
        clusters = model.fit_predict(scaled_data)
        df_clustered = data.copy()
        df_clustered["Cluster"] = clusters

        st.subheader("📊 Clustered Data")
        st.dataframe(df_clustered)

        # PCA 2D plot
        pca_2d = PCA(n_components=2)
        reduced_2d = pca_2d.fit_transform(scaled_data)

        fig2d, ax2d = plt.subplots()
        scatter = ax2d.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c=clusters, cmap="Set1")
        ax2d.set_title("2D Cluster Visualization")
        ax2d.set_xlabel("PCA 1")
        ax2d.set_ylabel("PCA 2")
        st.pyplot(fig2d)

        # PCA 3D plot
        pca_3d = PCA(n_components=3)
        reduced_3d = pca_3d.fit_transform(scaled_data)

        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2], c=clusters, cmap="Set1")
        ax3d.set_title("3D Cluster Visualization")
        ax3d.set_xlabel("PCA 1")
        ax3d.set_ylabel("PCA 2")
        ax3d.set_zlabel("PCA 3")
        st.pyplot(fig3d)

    else:
        st.warning("⚠️ Please select at least one feature to continue.")
