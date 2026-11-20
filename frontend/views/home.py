"""
Page 1 – Home Dashboard: overall project overview, real Retailrocket KPIs,
conversion funnel, and core supervised + unsupervised ML takeaways.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
import config

# =============================================================================
# 1. HEADER & PROJECT OVERVIEW
# =============================================================================
st.title(f"{config.PROJECT_ICON} {config.PROJECT_TITLE}")
st.caption(config.PROJECT_SUBTITLE)
st.write(config.PROJECT_DESCRIPTION)

try:
    ins = api_client.get_insights()
    schema = api_client.get_schema()
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

# =============================================================================
# 2. KEY E-COMMERCE & MODEL KPIs
# =============================================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw Clickstream Events", f"{ins.get('clean_event_rows', 0):,}", f"{ins.get('duplicates_removed', 0)} duplicates cleaned")
c2.metric("Unique Visitors", f"{ins.get('unique_visitors', 0):,}", f"{ins.get('unique_items', 0):,} products · {ins.get('total_categories', 0):,} categories")
c3.metric("Engineered Sessions", f"{ins.get('total_sessions', 0):,}", "30-min inactivity threshold")
c4.metric("Cart Abandonment Rate", f"{ins.get('cart_abandonment_rate_pct', 0):.1f}%", f"Overall session conversion {ins.get('session_conversion_rate_pct', 0):.2f}%")

m1, m2, m3, m4 = st.columns(4)
test_m = ins.get("test_metrics", {})
m1.metric("Modeling Dataset (clean.csv)", f"{ins.get('modeling_rows', 0):,} sessions", f"{ins.get('modeling_positive_rate_pct', 0):.2f}% purchase rate")
m2.metric("Best Classifier", ins.get("best_model") or ("Trained ✅" if schema["model_trained"] else "Not trained"), f"Test ROC-AUC: {test_m.get('roc_auc', '—')}")
m3.metric("Test F1 / Recall", f"{test_m.get('f1', '—')} / {test_m.get('recall', '—')}", f"Precision: {test_m.get('precision', '—')}")
m4.metric("Customer Segments (K-Means)", f"{len(ins.get('segments_overview', []))} Segments", f"Silhouette: {ins.get('clustering_silhouette', '—')}")

st.divider()

# =============================================================================
# 3. CONVERSION FUNNEL & CUSTOMER SEGMENT SNAPSHOT
# =============================================================================
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Session Conversion Funnel")
        funnel_df = pd.DataFrame(ins.get("session_funnel", []))
        if not funnel_df.empty:
            fig = px.funnel(
                funnel_df,
                x="count",
                y="stage",
                color_discrete_sequence=["#2F855A"],
            )
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Date coverage: **{ins.get('date_min', '—')}** to **{ins.get('date_max', '—')}** · "
            f"Views: **{ins.get('event_counts', {}).get('view', 0):,}** · "
            f"Add-to-carts: **{ins.get('event_counts', {}).get('addtocart', 0):,}** · "
            f"Transactions: **{ins.get('event_counts', {}).get('transaction', 0):,}**"
        )

with right:
    with st.container(border=True):
        st.subheader("Discovered Customer Segments (K-Means)")
        segs = ins.get("segments_overview", [])
        if segs:
            sdf = pd.DataFrame(segs)[["cluster_id", "segment_name", "size", "share_pct", "conversion_rate_pct"]]
            sdf.columns = ["Cluster", "Segment Name", "Customers", "Share (%)", "Conversion Rate (%)"]
            st.dataframe(sdf, use_container_width=True, hide_index=True)
        else:
            st.info("Run training on the Model Performance page to generate customer segments.")

# =============================================================================
# 4. SYSTEM ARCHITECTURE & NAVIGATION GUIDE
# =============================================================================
with st.container(border=True):
    st.subheader("System Architecture & CRISP-DM Workflow")
    st.markdown(
        """
        **Streamlit + React (`frontend/`)** ➜ HTTP/JSON (`api_client.py`) ➜ **FastAPI (`backend_fastapi/`)** ➜ **ML Pipeline (`ml/` & `models/`)**

        1. **Data Explorer** – Inspect cleaned session snapshots (`clean.csv`) and customer behavioral profiles (`customer_profiles.csv`).
        2. **Purchase Prediction** – Enter early-session customer behavior and predict conversion likelihood + business action.
        3. **Customer Segmentation** – Explore K-Means customer clusters, 2-D PCA projection, and interactive segment assignment.
        4. **Model Performance** – Compare 4 classifiers (5-fold CV), inspect test metrics/ROC/Confusion Matrix, and K-Means Elbow & Silhouette scores.
        5. **Prediction History & Customer Insights** – Audit SQLite-persisted predictions and actionable e-commerce conversion insights.
        """
    )
