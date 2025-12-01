#  Stage 3 — deployment with Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Step 1 — Load Artifacts
@st.cache_resource
def load_artifacts():
    path = "artifacts/stage2_model"
    imputer = joblib.load(os.path.join(path, "imputer.joblib"))
    scaler = joblib.load(os.path.join(path, "scaler.joblib"))
    model = joblib.load(os.path.join(path, "KMeans_model.joblib"))
    clustered_data = pd.read_csv(os.path.join(path, "clustered_data.csv"))
    cluster_profile = pd.read_csv(os.path.join(path, "cluster_profile_means.csv"))
    return imputer, scaler, model, clustered_data, cluster_profile

imputer, scaler, model, clustered_data, cluster_profile = load_artifacts()

# Step 2 — Sidebar Navigation
st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to:", [" Home", " Deep Segmentation", " Visual Analytics", " Export Results"])

st.sidebar.markdown("---")
st.sidebar.info("Developed by **Group 2** — Customer Segmentation Project (Stage 3)")

# Step 3 — Common Helper: Preprocess & Predict
def preprocess_and_predict(data):
    data = data.copy()

    # Feature engineering (same as Stage 2)
    data['Customer_Age'] = 2025 - data['Year_Birth']
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], errors='coerce')
    data['Tenure_days'] = (pd.Timestamp('2025-10-31') - data['Dt_Customer']).dt.days
    spend_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    data['TotalSpend'] = data[spend_cols].sum(axis=1)

    selected_features = [
        'Income','Recency','Customer_Age','Tenure_days','TotalSpend',
        'MntWines','MntFruits','MntMeatProducts','MntFishProducts',
        'MntSweetProducts','MntGoldProds','NumDealsPurchases',
        'NumWebPurchases','NumStorePurchases','NumWebVisitsMonth',
        'Complain','Response'
    ]

    X = data[selected_features]
    X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)
    X_scaled = scaler.transform(X_imputed)
    labels = model.predict(X_scaled)

    data['Predicted_Cluster'] = labels
    labels_map = {
        0: "High-Value & Active",
        1: "Medium-High Value",
        2: "Medium Value",
        3: "Low-Value / At-Risk"
    }
    data['Cluster_Label'] = data['Predicted_Cluster'].map(labels_map)
    return data


#  Page 1 — Home
if page == " Home":
    st.title(" Customer Segmentation Overview")

    uploaded_file = st.file_uploader("Upload Customer File (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            new_data = pd.read_csv(uploaded_file)
        else:
            new_data = pd.read_excel(uploaded_file)
        st.success(" File uploaded successfully!")

        processed_data = preprocess_and_predict(new_data)

        st.write("### Preview of Clustered Data")
        st.dataframe(processed_data.head())

        # Key metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(processed_data))
        col2.metric("Unique Segments", processed_data['Predicted_Cluster'].nunique())
        top_cluster = processed_data['Cluster_Label'].mode()[0]
        col3.metric("Dominant Segment", top_cluster)

        st.session_state["processed_data"] = processed_data

    else:
        st.info("Please upload a dataset to view segmentation results.")

#  Page 2 — Deep Segmentation
elif page == " Deep Segmentation":
    st.title(" Deep Segmentation Insights")

    if "processed_data" not in st.session_state:
        st.warning(" Please upload data from the Home page first.")
        st.stop()

    data = st.session_state["processed_data"]

    # Segment filter
    clusters = sorted(data['Cluster_Label'].unique())
    selected_cluster = st.selectbox("Select a Segment to Explore", clusters)

    seg_data = data[data['Cluster_Label'] == selected_cluster]

    st.subheader(f" Segment Summary: {selected_cluster}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", len(seg_data))
    col2.metric("Avg Income", f"{seg_data['Income'].mean():,.0f}")
    col3.metric("Avg Total Spend", f"{seg_data['TotalSpend'].mean():,.0f}")

    # Spending breakdown
    st.write("###  Average Spending Breakdown")
    spend_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    avg_spend = seg_data[spend_cols].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=avg_spend.index, y=avg_spend.values, palette="Blues_r", ax=ax)
    plt.xticks(rotation=30)
    plt.title("Average Spend by Category")
    st.pyplot(fig)

    # Recency distribution
    st.write("### ⏱ Recency Distribution")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.histplot(seg_data['Recency'], bins=20, kde=True, color="teal", ax=ax2)
    plt.title("Recency Distribution for Selected Segment")
    st.pyplot(fig2)

#  Page 3 — Visual Analytics (Enhanced & Fixed)
elif page == " Visual Analytics":
    st.title(" Overall Cluster Analytics")

    if "processed_data" not in st.session_state:
        st.warning(" Please upload data from the Home page first.")
        st.stop()

    data = st.session_state["processed_data"]

    #  Use same features from Stage 2 model training
    selected_features = [
        "Income", "Recency", "Customer_Age", "Tenure_days", "TotalSpend",
        "MntWines","MntFruits","MntMeatProducts","MntFishProducts",
        "MntSweetProducts","MntGoldProds",
        "NumDealsPurchases","NumWebPurchases","NumStorePurchases",
        "NumWebVisitsMonth","Complain","Response"
    ]
    available_features = [f for f in selected_features if f in data.columns]

    # 🔹 Cluster Distribution
    st.write("### 🔍 Cluster Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    order = data['Cluster_Label'].value_counts().index
    sns.countplot(
        x="Cluster_Label",
        data=data,
        order=order,
        palette="tab10",
        ax=ax
    )
    plt.title("Number of Customers per Segment", fontsize=13)
    plt.xlabel("Cluster Label")
    plt.ylabel("Customer Count")
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', fontsize=9)
    st.pyplot(fig)


    #  PCA Visualization
    st.write("###  Cluster Separation (PCA Projection)")

    try:
        X = data[available_features]
        X_imputed = pd.DataFrame(imputer.transform(X), columns=available_features)
        X_scaled = scaler.transform(X_imputed)

        proj = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=proj[:, 0],
            y=proj[:, 1],
            hue=data["Cluster_Label"],
            palette="tab10",
            s=60,
            alpha=0.8,
            ax=ax2
        )
        plt.title("Customer Clusters (2D PCA Projection)", fontsize=13)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Cluster", loc="best")
        st.pyplot(fig2)

        st.info("This visualization projects high-dimensional customer data into 2D using PCA to show how distinct the clusters are.")

    except Exception as e:
        st.error(f"⚠️ PCA visualization could not be generated: {e}")


#  Page 4 — Export Results
elif page == " Export Results":
    st.title(" Export Clustered Data")

    if "processed_data" not in st.session_state:
        st.warning(" Please upload data from the Home page first.")
        st.stop()

    data = st.session_state["processed_data"]
    st.dataframe(data.head())

    st.download_button(
        label="⬇ Download Full Clustered Data (CSV)",
        data=data.to_csv(index=False),
        file_name="customer_segments.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("© 2025 — Developed by Group 2 | Customer Segmentation")
