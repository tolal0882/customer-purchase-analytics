"""
Page 2 – Data Explorer: inspect processed datasets, apply interactive filters,
view descriptive statistics, and explore feature distributions vs target.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client

st.title("📊 Data Explorer")
st.caption("Explore the leakage-safe Purchase Prediction dataset (`clean.csv`) and Customer Segmentation profiles (`customer_profiles.csv`).")

# =============================================================================
# 1. DATASET SELECTOR & ROW LIMIT
# =============================================================================
c_ds, c_lim = st.columns([2.2, 1])
ds_choice = c_ds.radio(
    "Select Dataset",
    [
        "Purchase Prediction Snapshots (clean.csv)",
        "Customer Behavioral Profiles (customer_profiles.csv)",
        "Product Category Tree (category_tree.csv)",
    ],
    horizontal=True,
)
ds_key = "modeling" if ds_choice.startswith("Purchase") else ("customers" if ds_choice.startswith("Customer") else "categories")
limit = int(c_lim.selectbox("Sample rows to load", [500, 1000, 3000, 5000], index=1))

try:
    data = api_client.get_data(limit=limit, dataset=ds_key)
    schema = api_client.get_schema()
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

df = pd.DataFrame(data["rows"], columns=data["columns"])
if ds_key == "categories":
    st.caption(f"Showing **{len(df):,}** of **{data['total_rows']:,}** categories from `data/raw/category_tree.csv` (`{int(df['parentid'].isna().sum())}` root categories).")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.stop()
target = schema["target"] if ds_key == "modeling" else "converted"

# =============================================================================
# 2. INTERACTIVE FILTER & DATA PREVIEW
# =============================================================================
with st.container(border=True):
    f1, f2 = st.columns([1, 2])
    target_filter = f1.multiselect(
        f"Filter by `{target}`",
        options=sorted(df[target].dropna().astype(str).unique()),
        default=sorted(df[target].dropna().astype(str).unique()),
    )
    cols_to_show = f2.multiselect("Columns to display", list(df.columns), default=list(df.columns))

filtered = df[df[target].astype(str).isin(target_filter)] if target_filter else df
st.caption(f"Showing **{len(filtered):,}** filtered rows (out of **{data['total_rows']:,}** total rows in `{ds_key}` dataset).")
st.dataframe(filtered[cols_to_show].head(100), use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download Filtered Sample (CSV)",
    filtered[cols_to_show].to_csv(index=False).encode("utf-8"),
    file_name=f"retailrocket_{ds_key}_sample.csv",
    mime="text/csv",
)

# =============================================================================
# 3. SUMMARY STATISTICS & TARGET DISTRIBUTION
# =============================================================================
num_cols = [c for c in filtered.select_dtypes(include="number").columns if c not in ("session_id", "visitorid", "session_start_ts")]
cat_cols = [c for c in filtered.columns if c not in filtered.select_dtypes(include="number").columns and c != target]

c1, c2 = st.columns([1.4, 1])
with c1:
    with st.container(border=True):
        st.subheader("Descriptive Statistics")
        st.dataframe(filtered[num_cols].describe().T.round(2), use_container_width=True)

with c2:
    with st.container(border=True):
        st.subheader(f"Target Distribution (`{target}`)")
        vc = filtered[target].value_counts().reset_index()
        vc.columns = [target, "count"]
        vc["share_pct"] = (vc["count"] / max(len(filtered), 1) * 100).round(2)
        fig = px.bar(
            vc, x=target, y="count", color=target, text="share_pct",
            color_discrete_map={"Yes": "#2F855A", "No": "#6B7C74"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 4. INTERACTIVE FEATURE VS TARGET VISUALIZATION
# =============================================================================
with st.container(border=True):
    st.subheader("Interactive Feature Exploration")
    explore_cols = num_cols + cat_cols
    col = st.selectbox("Select a feature to visualize against conversion", explore_cols)
    p1, p2 = st.columns(2)
    if col in num_cols:
        with p1:
            fig_hist = px.histogram(
                filtered, x=col, color=target, barmode="overlay", nbins=25,
                color_discrete_map={"Yes": "#2F855A", "No": "#94A3B8"},
                title=f"Distribution of `{col}` by `{target}`",
            )
            fig_hist.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)
        with p2:
            fig_box = px.box(
                filtered, x=target, y=col, color=target,
                color_discrete_map={"Yes": "#2F855A", "No": "#94A3B8"},
                title=f"Box Plot of `{col}` by `{target}`",
            )
            fig_box.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        ct = (
            filtered.groupby([col, target])
            .size()
            .reset_index(name="count")
        )
        fig_cat = px.bar(
            ct, x=col, y="count", color=target, barmode="group",
            color_discrete_map={"Yes": "#2F855A", "No": "#94A3B8"},
            title=f"`{col}` vs `{target}`",
        )
        fig_cat.update_layout(height=320)
        st.plotly_chart(fig_cat, use_container_width=True)
