import plotly.express as px
import streamlit as st
import numpy as np

from utils.data_loader import load_csv, load_image_base64
from utils.preprocessing import preprocess_dataframe
from utils.evaluation import get_metrics

from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")

encoded_logo = load_image_base64("etc/logo.png")

st.markdown(
    f"""
    <div style="text-align: left;">
        <img src="data:image/png;base64,{encoded_logo}" alt="Logo" style="width:600px;"><br>
        <p style="font-size:18px;"> This portal allows users to easily explore and visualize data patterns using clustering algorithms.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Step 1 - Upload CSV
st.subheader("Step 1 - Select database")
uploaded_file = st.file_uploader("Database in .csv format:", type="csv")

if uploaded_file is None:
    st.session_state.df = None
else:
    df = load_csv(uploaded_file)
    st.session_state.df = df

# Step 2 - Select clustering algorithm
st.subheader("Step 2 - Select clustering algorithm")

algorithm = st.selectbox(
    "Choose a clustering algorithm:",
    ("KMeans", "DBSCAN", "Mean Shift", "Spectral Clustering")
)

st.session_state.algorithm = algorithm

if algorithm == "KMeans":
    n_clusters = st.slider("Number of clusters (K):", min_value=2, max_value=10, value=3)
    st.session_state.n_clusters = n_clusters

elif algorithm == "DBSCAN":
    eps = st.slider("Epsilon (radius):", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    min_samples = st.slider("Minimum number of samples per cluster:", min_value=1, max_value=20, value=5)
    st.session_state.eps = eps
    st.session_state.min_samples = min_samples

elif algorithm == "Mean Shift":
    bandwidth = st.number_input("Bandwidth (leave 0 for automatic estimation):", min_value=0.0, value=0.0, step=0.1)
    st.session_state.bandwidth = bandwidth

elif algorithm == "Spectral Clustering":
    n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=3)
    st.session_state.n_clusters = n_clusters

# Step 3 - Choose visualization type
st.subheader("Step 3 - Choose dimensionality reduction")

vis_type = st.radio("Select PCA projection:", ["2D", "3D"])
st.session_state.vis_type = vis_type

# Step 4 - Apply clustering and visualize
st.subheader("Step 4 - Start clustering and visualization")

if st.button("Start"):
    if st.session_state.df is not None:   
        df = st.session_state.df.copy()
        num_df = df.select_dtypes(include=np.number)

        if num_df.shape[1] >= 2:          
            # Preprocessing dataframe
            df, data_scaled, valid_indices = preprocess_dataframe(df)

            # Fit clustering
            if algorithm == "KMeans":
                model = KMeans(n_clusters=st.session_state.n_clusters, random_state=42)
                labels = model.fit_predict(data_scaled)

            elif algorithm == "DBSCAN":
                model = DBSCAN(eps=st.session_state.eps, min_samples=st.session_state.min_samples)
                labels = model.fit_predict(data_scaled)

            elif algorithm == "Mean Shift":
                if st.session_state.bandwidth == 0:
                    model = MeanShift()
                else:
                    model = MeanShift(bandwidth=st.session_state.bandwidth)
                labels = model.fit_predict(data_scaled)

            elif algorithm == "Spectral Clustering":
                model = SpectralClustering(
                    n_clusters=st.session_state.n_clusters,
                    affinity='nearest_neighbors',
                    assign_labels='kmeans',
                    random_state=42
                )
                labels = model.fit_predict(data_scaled)

            df['Cluster'] = labels.astype(str)  # Convert to string for Plotly

            # PCA for 2D or 3D
            if vis_type == "2D":
                pca = PCA(n_components=2)
                components = pca.fit_transform(data_scaled)
                df['PC1'] = components[:, 0]
                df['PC2'] = components[:, 1]

                st.write("#### Cluster visualization (PCA 2D)")
                fig = px.scatter(
                    df,
                    x="PC1",
                    y="PC2",
                    color="Cluster",
                    hover_data=df.columns
                )
                st.plotly_chart(fig, use_container_width=True)

            elif vis_type == "3D":
                pca = PCA(n_components=3)
                components = pca.fit_transform(data_scaled)
                df['PC1'] = components[:, 0]
                df['PC2'] = components[:, 1]
                df['PC3'] = components[:, 2]

                st.write("#### Cluster visualization (PCA 3D)")
                fig = px.scatter_3d(
                    df,
                    x="PC1",
                    y="PC2",
                    z="PC3",
                    color="Cluster",
                    hover_data=df.columns
                )
                st.plotly_chart(fig, use_container_width=True)

            metrics = get_metrics(data_scaled, labels)

            st.write("#### Clustering Evaluation Metrics")

            if metrics is None:
                st.warning("Clustering has less than 2 valid clusters. Metrics cannot be computed.")
            else:
                metric_explanations = {
                    "Silhouette Score": "Ranges from -1 to 1. Higher is better. Indicates how well each point fits within its cluster.",
                    "Calinski-Harabasz Index": "No fixed upper bound. Higher is better. Measures ratio of inter- to intra-cluster dispersion.",
                    "Davies-Bouldin Index": "Lower is better. Starts from 0. Measures average similarity between clusters."
                }

                for name, value in metrics.items():
                    st.markdown(f"**{name}:** `{value:.3f}` {metric_explanations[name]}")

            # Show dataframe
            st.write("#### Data with cluster labels")
            st.dataframe(df)
        else:
            st.warning("Please upload/select a dataset with at least two numeric features.")
    else:
        st.info("Please upload a CSV file first.")

# Rodapé
st.markdown(
    """
    <hr style="margin-top: 40px; margin-bottom: 10px;"/>

    <div style="text-align: center; font-size: 0.85em; color: gray;">
        © 2025 Data Inception. For educational and research purposes. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)