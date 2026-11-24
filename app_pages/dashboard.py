"""
PAGE 1 — Dashboard (Section 8):
  - Filters: Month, Visitor Type, Weekend, Revenue
  - KPI Cards (6):
      1. Total Sessions
      2. Purchase Sessions
      3. Non-Purchase Sessions
      4. Purchase Rate
      5. Average Product-Related Pages
      6. Average Product-Related Duration
  - Charts (6):
      1. Purchase vs. No Purchase distribution
      2. Purchase rate by month
      3. Purchase rate by visitor type
      4. Purchase rate by weekend
      5. Product browsing behavior
      6. Session behavior overview
  - All KPIs and charts update dynamically with the active filters.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import data_loader, data_validation, ui

ui.inject_css()
df_raw = data_loader.require_dataset()

ui.page_header(
    title="Online Shopping Behavior Dashboard",
    subtitle="Interactive overview of the UCI Online Shoppers Purchasing Intention Dataset (12,330 sessions across 17 features and Revenue outcome).",
    category="Page 1 · Executive Overview",
    badges=[
        (f"Total Dataset: {len(df_raw):,} Sessions", "ok"),
        ("17 Input Features + Revenue Target", "info"),
        ("CC BY 4.0 · UCI Repository", "neutral"),
    ],
)

# =============================================================================
# 1. DYNAMIC DASHBOARD FILTERS (Month, VisitorType, Weekend, Revenue)
# =============================================================================
available_months = [m for m in data_validation.MONTH_ORDER if m in df_raw["Month"].unique()]
available_visitors = sorted(df_raw["VisitorType"].astype(str).unique().tolist())

with st.container(border=True):
    st.markdown('<div class="card-title">Interactive Session Filters</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1.5, 1.3, 1, 1])
    sel_months = f1.multiselect("Month", options=available_months, default=available_months)
    sel_visitors = f2.multiselect("Visitor Type", options=available_visitors, default=available_visitors)
    sel_weekend = f3.multiselect(
        "Weekend",
        options=[False, True],
        default=[False, True],
        format_func=lambda x: "Weekend (True)" if x else "Weekday (False)",
    )
    sel_revenue = f4.multiselect(
        "Revenue (Purchase Outcome)",
        options=[False, True],
        default=[False, True],
        format_func=lambda x: "Purchase" if x else "No Purchase",
    )

mask = (
    df_raw["Month"].isin(sel_months)
    & df_raw["VisitorType"].astype(str).isin(sel_visitors)
    & df_raw["Weekend"].astype(bool).isin(sel_weekend)
    & df_raw["Revenue"].astype(bool).isin(sel_revenue)
)
df = df_raw.loc[mask].copy()

if df.empty:
    ui.empty_state(
        "No sessions match the selected filters",
        "Adjust one or more filter selections above (Month, Visitor Type, Weekend, or Revenue) to view metrics and charts.",
        "🔍",
    )
    st.stop()

# =============================================================================
# 2. SIX DYNAMIC KPI CARDS
# =============================================================================
total_sessions = len(df)
rev_bool = df["Revenue"].astype(bool)
purchase_sessions = int(rev_bool.sum())
non_purchase_sessions = int(total_sessions - purchase_sessions)
purchase_rate_pct = purchase_sessions / max(total_sessions, 1) * 100.0
avg_prod_pages = float(df["ProductRelated"].mean())
avg_prod_duration = float(df["ProductRelated_Duration"].mean())

ui.kpi_row(
    [
        ("Total Sessions", f"{total_sessions:,}", f"{total_sessions / len(df_raw) * 100:.1f}% of full dataset"),
        ("Purchase Sessions", f"{purchase_sessions:,}", "Revenue = True (1)"),
        ("Non-Purchase Sessions", f"{non_purchase_sessions:,}", "Revenue = False (0)"),
        ("Purchase Rate", f"{purchase_rate_pct:.2f}%", f"1 in {max(total_sessions / max(purchase_sessions, 1), 1):.1f} sessions convert"),
        ("Average Product-Related Pages", f"{avg_prod_pages:.2f}", f"Median: {df['ProductRelated'].median():.0f} pages"),
        ("Average Product-Related Duration", f"{avg_prod_duration:,.1f} s", f"{avg_prod_duration / 60:.1f} min average"),
    ],
    per_row=3,
)

# =============================================================================
# 3. SIX INTERACTIVE PLOTLY CHARTS
# =============================================================================
c1, c2 = st.columns(2)

# Chart 1: Purchase vs. No Purchase distribution
with c1, ui.chart_card("1. Purchase vs. No Purchase Distribution", "Session count and percentage breakdown by Revenue outcome"):
    dist_df = pd.DataFrame([
        {"Outcome": "No Purchase (False)", "Sessions": non_purchase_sessions, "Share (%)": round(100 - purchase_rate_pct, 2)},
        {"Outcome": "Purchase (True)", "Sessions": purchase_sessions, "Share (%)": round(purchase_rate_pct, 2)},
    ])
    fig_dist = px.bar(
        dist_df,
        x="Outcome",
        y="Sessions",
        color="Outcome",
        text="Share (%)",
        color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
    )
    fig_dist.update_traces(texttemplate="%{y:,} (%{text:.1f}%)", textposition="outside")
    ui.apply_plotly_theme(fig_dist, height=310)
    fig_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

# Chart 2: Purchase rate by month
with c2, ui.chart_card("2. Purchase Rate by Month", "Percentage of sessions resulting in Revenue = True across calendar months"):
    m_grp = (
        df.assign(rev_num=df["Revenue"].astype(bool).astype(int))
        .groupby("Month", as_index=False)
        .agg(sessions=("rev_num", "size"), purchases=("rev_num", "sum"), purchase_rate=("rev_num", "mean"))
    )
    m_grp["purchase_rate_pct"] = (m_grp["purchase_rate"] * 100).round(2)
    m_grp["Month"] = pd.Categorical(m_grp["Month"], categories=data_validation.MONTH_ORDER, ordered=True)
    m_grp = m_grp.sort_values("Month")
    fig_month = px.bar(
        m_grp,
        x="Month",
        y="purchase_rate_pct",
        text="purchase_rate_pct",
        hover_data=["sessions", "purchases"],
        color="purchase_rate_pct",
        color_continuous_scale=["#A7C957", "#2F855A", "#153E2D"],
        labels={"purchase_rate_pct": "Purchase Rate (%)"},
    )
    fig_month.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    ui.apply_plotly_theme(fig_month, height=310)
    fig_month.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_month, use_container_width=True)

c3, c4 = st.columns(2)

# Chart 3: Purchase rate by visitor type
with c3, ui.chart_card("3. Purchase Rate by Visitor Type", "Conversion rate across Returning_Visitor, New_Visitor, and Other"):
    v_grp = (
        df.assign(rev_num=df["Revenue"].astype(bool).astype(int))
        .groupby("VisitorType", as_index=False)
        .agg(sessions=("rev_num", "size"), purchases=("rev_num", "sum"), purchase_rate=("rev_num", "mean"))
    )
    v_grp["purchase_rate_pct"] = (v_grp["purchase_rate"] * 100).round(2)
    fig_vis = px.bar(
        v_grp,
        x="VisitorType",
        y="purchase_rate_pct",
        color="VisitorType",
        text="purchase_rate_pct",
        hover_data=["sessions", "purchases"],
        color_discrete_sequence=["#2F855A", "#153E2D", "#A7C957"],
        labels={"purchase_rate_pct": "Purchase Rate (%)"},
    )
    fig_vis.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    ui.apply_plotly_theme(fig_vis, height=310)
    fig_vis.update_layout(showlegend=False)
    st.plotly_chart(fig_vis, use_container_width=True)

# Chart 4: Purchase rate by weekend
with c4, ui.chart_card("4. Purchase Rate by Weekend vs. Weekday", "Comparison of purchase conversion on weekends vs. weekdays"):
    w_grp = (
        df.assign(
            Weekend_Label=df["Weekend"].astype(bool).map({False: "Weekday (False)", True: "Weekend (True)"}),
            rev_num=df["Revenue"].astype(bool).astype(int),
        )
        .groupby("Weekend_Label", as_index=False)
        .agg(sessions=("rev_num", "size"), purchases=("rev_num", "sum"), purchase_rate=("rev_num", "mean"))
    )
    w_grp["purchase_rate_pct"] = (w_grp["purchase_rate"] * 100).round(2)
    fig_wk = px.bar(
        w_grp,
        x="Weekend_Label",
        y="purchase_rate_pct",
        color="Weekend_Label",
        text="purchase_rate_pct",
        hover_data=["sessions", "purchases"],
        color_discrete_map={"Weekday (False)": "#2F855A", "Weekend (True)": "#153E2D"},
        labels={"Weekend_Label": "Day Type", "purchase_rate_pct": "Purchase Rate (%)"},
    )
    fig_wk.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    ui.apply_plotly_theme(fig_wk, height=310)
    fig_wk.update_layout(showlegend=False)
    st.plotly_chart(fig_wk, use_container_width=True)

c5, c6 = st.columns(2)

# Chart 5: Product browsing behavior
with c5, ui.chart_card("5. Product Browsing Behavior by Purchase Outcome", "Average Product-Related pages and duration (minutes) by Revenue"):
    p_grp = (
        df.assign(
            Outcome=df["Revenue"].astype(bool).map({False: "No Purchase (False)", True: "Purchase (True)"}),
            Product_Duration_Min=df["ProductRelated_Duration"] / 60.0,
        )
        .groupby("Outcome", as_index=False)
        .agg(
            Avg_Product_Pages=("ProductRelated", "mean"),
            Avg_Product_Duration_Min=("Product_Duration_Min", "mean"),
        )
        .round(2)
    )
    p_long = p_grp.melt("Outcome", var_name="Metric", value_name="Average")
    p_long["Metric"] = p_long["Metric"].map({
        "Avg_Product_Pages": "Avg Product Pages Viewed",
        "Avg_Product_Duration_Min": "Avg Product Duration (Minutes)",
    })
    fig_prod = px.bar(
        p_long,
        x="Metric",
        y="Average",
        color="Outcome",
        barmode="group",
        text="Average",
        color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
    )
    fig_prod.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    ui.apply_plotly_theme(fig_prod, height=320)
    st.plotly_chart(fig_prod, use_container_width=True)

# Chart 6: Session behavior overview across page types & interaction metrics
with c6, ui.chart_card("6. Session Behavior Overview", "Mean page visits and interaction rates (Bounce, Exit, PageValues) by outcome"):
    s_grp = (
        df.assign(Outcome=df["Revenue"].astype(bool).map({False: "No Purchase (False)", True: "Purchase (True)"}))
        .groupby("Outcome", as_index=False)
        .agg(
            Administrative=("Administrative", "mean"),
            Informational=("Informational", "mean"),
            PageValues=("PageValues", "mean"),
            BounceRate_Pct=("BounceRates", lambda s: s.mean() * 100),
            ExitRate_Pct=("ExitRates", lambda s: s.mean() * 100),
        )
        .round(2)
    )
    s_long = s_grp.melt("Outcome", var_name="Behavioral Metric", value_name="Mean Value")
    fig_beh = px.bar(
        s_long,
        x="Behavioral Metric",
        y="Mean Value",
        color="Outcome",
        barmode="group",
        text="Mean Value",
        color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
    )
    fig_beh.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    ui.apply_plotly_theme(fig_beh, height=320)
    st.plotly_chart(fig_beh, use_container_width=True)
