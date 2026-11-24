"""
PAGE 2 — Dataset Explorer & EDA (Section 8):
  - Dataset Explorer:
      * Data preview, search and filtering, column selection, data types,
        dataset dimensions, descriptive statistics, missing values,
        duplicate records, CSV export.
  - Exploratory Data Analysis (4 Sections):
      * A. Purchase Analysis
      * B. Browsing Behavior
      * C. Website Interaction
      * D. Feature Relationships
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import data_loader, data_validation, ui

ui.inject_css()
df_raw = data_loader.require_dataset()
val_report = data_validation.validate_dataset(df_raw)

ui.page_header(
    title="Dataset Explorer & Exploratory Data Analysis",
    subtitle="Inspect the raw UCI Online Shoppers dataset, filter and export records, and explore purchase, browsing, interaction, and correlation patterns.",
    category="Page 2 · Data Understanding",
    badges=[
        (f"{val_report['n_rows']:,} Rows × {val_report['n_cols']} Columns", "ok"),
        (f"Missing Values: {val_report['n_missing_total']}", "info"),
        (f"Exact Duplicates: {val_report['n_duplicates']}", "warn"),
    ],
)

main_tab_exp, main_tab_eda = st.tabs([
    "🗂️ 1. Interactive Dataset Explorer",
    "📊 2. Exploratory Data Analysis (A · B · C · D)",
])

# =============================================================================
# PART 1: DATASET EXPLORER
# =============================================================================
with main_tab_exp:
    ui.kpi_row(
        [
            ("Dataset Dimensions", f"{val_report['n_rows']:,} × {val_report['n_cols']}", "17 input features + Revenue"),
            ("Missing Values", f"{val_report['n_missing_total']:,}", "0.00% missing across all 18 columns"),
            ("Exact Duplicate Rows", f"{val_report['n_duplicates']:,}", f"{val_report['duplicate_pct']:.2f}% of sessions (not silently deleted)"),
        ],
        per_row=3,
    )

    with ui.chart_card("Search, Filter & Select Columns", "Filter sessions by text search, Month, VisitorType, Revenue, and numerical ranges"):
        f1, f2, f3, f4 = st.columns([1.4, 1.2, 1.2, 1])
        search_txt = f1.text_input("Search across rows", placeholder="e.g. Returning_Visitor or Nov")
        m_opts = [m for m in data_validation.MONTH_ORDER if m in df_raw["Month"].unique()]
        m_filt = f2.multiselect("Filter Month", options=m_opts, default=m_opts, key="exp_m")
        v_opts = sorted(df_raw["VisitorType"].astype(str).unique().tolist())
        v_filt = f3.multiselect("Filter VisitorType", options=v_opts, default=v_opts, key="exp_v")
        r_filt = f4.multiselect(
            "Filter Revenue",
            options=[False, True],
            default=[False, True],
            format_func=lambda x: "Purchase (True)" if x else "No Purchase (False)",
            key="exp_r",
        )

        c_col, c_sort, c_ord, c_pg = st.columns([2.2, 1.1, 0.9, 0.8])
        selected_cols = c_col.multiselect("Selected Columns", options=list(df_raw.columns), default=list(df_raw.columns))
        sort_col = c_sort.selectbox("Sort by Column", options=list(df_raw.columns), index=8)  # PageValues
        sort_asc = c_ord.selectbox("Order", ["Descending", "Ascending"]) == "Ascending"
        page_size = int(c_pg.selectbox("Rows / page", [25, 50, 100, 250], index=1))

    filtered = df_raw[
        df_raw["Month"].isin(m_filt)
        & df_raw["VisitorType"].astype(str).isin(v_filt)
        & df_raw["Revenue"].astype(bool).isin(r_filt)
    ].copy()

    if search_txt.strip():
        q = search_txt.strip()
        row_match = filtered.astype(str).apply(lambda s: s.str.contains(q, case=False, na=False, regex=False)).any(axis=1)
        filtered = filtered.loc[row_match]

    filtered = filtered.sort_values(sort_col, ascending=sort_asc, kind="stable").reset_index(drop=True)
    cols_to_show = selected_cols or list(df_raw.columns)

    total_pages = max(1, int(np.ceil(len(filtered) / page_size)))
    p_col, dl_col = st.columns([1, 2])
    page_num = int(p_col.number_input(f"Page (1 to {total_pages})", min_value=1, max_value=total_pages, value=1))
    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered))

    dl_col.write("")
    dl_col.download_button(
        f"⬇️ Export Filtered Dataset ({len(filtered):,} rows CSV)",
        filtered[cols_to_show].to_csv(index=False).encode("utf-8"),
        file_name="online_shoppers_filtered_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with ui.chart_card(f"Data Preview — Showing rows {start_idx + 1:,} to {end_idx:,} of {len(filtered):,} matching sessions"):
        st.dataframe(filtered.iloc[start_idx:end_idx][cols_to_show], use_container_width=True, hide_index=True)

    d1, d2 = st.columns([1.35, 1])
    with d1, ui.chart_card("Descriptive Statistics (Numerical & Coded Features)", "Computed on the currently filtered dataset"):
        num_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            desc = filtered[num_cols].describe().T
            desc["median"] = filtered[num_cols].median()
            desc["IQR"] = filtered[num_cols].quantile(0.75) - filtered[num_cols].quantile(0.25)
            st.dataframe(desc[["count", "mean", "std", "min", "median", "IQR", "max"]].round(4), use_container_width=True)

    with d2, ui.chart_card("Column Data Types, Missing Values & Unique Counts", "Full schema inspection across all 18 columns"):
        st.dataframe(pd.DataFrame(val_report["column_summary"]), use_container_width=True, hide_index=True)

    with st.expander(f"🔎 Inspect Exact Duplicate Records ({val_report['n_duplicates']} duplicate rows detected in raw CSV)"):
        ui.note(
            "Per methodological rules, these <b>125 exact duplicate records</b> are never silently deleted from the raw file. "
            "You can inspect them below and configure explicit duplicate handling on <b>Page 3 (Data Preprocessing)</b>."
        )
        dup_rows = df_raw[df_raw.duplicated(keep=False)].sort_values(by=["ProductRelated", "BounceRates", "ExitRates"])
        st.dataframe(dup_rows.head(100), use_container_width=True, hide_index=True)

# =============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS (A, B, C, D)
# =============================================================================
with main_tab_eda:
    ui.note(
        "<b>Methodological Note:</b> All charts below are computed directly from the 12,330 sessions in "
        "<code>online_shoppers_intention.csv</code> and describe <b>observational associations</b> across sessions "
        "(not causal effects or long-term customer histories)."
    )

    eda_a, eda_b, eda_c, eda_d = st.tabs([
        "A · Purchase Analysis",
        "B · Browsing Behavior",
        "C · Website Interaction",
        "D · Feature Relationships",
    ])

    df_eda = df_raw.copy()
    df_eda["Outcome"] = df_eda["Revenue"].astype(bool).map({False: "No Purchase (False)", True: "Purchase (True)"})
    df_eda["rev_int"] = df_eda["Revenue"].astype(bool).astype(int)

    # -------------------------------------------------------------------------
    # A. PURCHASE ANALYSIS
    # -------------------------------------------------------------------------
    with eda_a:
        a1, a2 = st.columns(2)
        with a1, ui.chart_card("A1. Purchase Outcome Distribution", "Class balance across all 12,330 shopping sessions"):
            vc = df_eda["Outcome"].value_counts().reset_index()
            vc.columns = ["Outcome", "Sessions"]
            vc["Share (%)"] = (vc["Sessions"] / len(df_eda) * 100).round(2)
            fig_a1 = px.pie(
                vc,
                names="Outcome",
                values="Sessions",
                hole=0.45,
                color="Outcome",
                color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
            )
            fig_a1.update_traces(textinfo="label+percent+value")
            ui.apply_plotly_theme(fig_a1, height=320)
            st.plotly_chart(fig_a1, use_container_width=True)

        with a2, ui.chart_card("A2. Purchase Rate & Volume by Month", "Seasonal variation in session volume and purchase conversion rate"):
            m_df = (
                df_eda.groupby("Month", as_index=False)
                .agg(Sessions=("rev_int", "size"), Purchases=("rev_int", "sum"), Purchase_Rate=("rev_int", "mean"))
            )
            m_df["Purchase_Rate_Pct"] = (m_df["Purchase_Rate"] * 100).round(2)
            m_df["Month"] = pd.Categorical(m_df["Month"], categories=data_validation.MONTH_ORDER, ordered=True)
            m_df = m_df.sort_values("Month")
            fig_a2 = px.bar(
                m_df,
                x="Month",
                y="Purchase_Rate_Pct",
                text="Purchase_Rate_Pct",
                hover_data=["Sessions", "Purchases"],
                color="Purchase_Rate_Pct",
                color_continuous_scale=["#A7C957", "#2F855A", "#153E2D"],
                labels={"Purchase_Rate_Pct": "Purchase Rate (%)"},
            )
            fig_a2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            ui.apply_plotly_theme(fig_a2, height=320)
            fig_a2.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_a2, use_container_width=True)

        a3, a4 = st.columns(2)
        with a3, ui.chart_card("A3. Purchase Rate by Visitor Type", "New_Visitor sessions show a higher purchase rate (24.91%) than Returning_Visitor (13.93%)"):
            v_df = (
                df_eda.groupby("VisitorType", as_index=False)
                .agg(Sessions=("rev_int", "size"), Purchases=("rev_int", "sum"), Purchase_Rate=("rev_int", "mean"))
            )
            v_df["Purchase_Rate_Pct"] = (v_df["Purchase_Rate"] * 100).round(2)
            fig_a3 = px.bar(
                v_df,
                x="VisitorType",
                y="Purchase_Rate_Pct",
                color="VisitorType",
                text="Purchase_Rate_Pct",
                hover_data=["Sessions", "Purchases"],
                color_discrete_sequence=["#2F855A", "#153E2D", "#A7C957"],
                labels={"Purchase_Rate_Pct": "Purchase Rate (%)"},
            )
            fig_a3.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            ui.apply_plotly_theme(fig_a3, height=310)
            fig_a3.update_layout(showlegend=False)
            st.plotly_chart(fig_a3, use_container_width=True)

        with a4, ui.chart_card("A4. Purchase Rate by Weekend Status", "Comparison between weekend (17.40%) and weekday (14.89%) sessions"):
            w_df = (
                df_eda.assign(Weekend_Status=df_eda["Weekend"].astype(bool).map({False: "Weekday (False)", True: "Weekend (True)"}))
                .groupby("Weekend_Status", as_index=False)
                .agg(Sessions=("rev_int", "size"), Purchases=("rev_int", "sum"), Purchase_Rate=("rev_int", "mean"))
            )
            w_df["Purchase_Rate_Pct"] = (w_df["Purchase_Rate"] * 100).round(2)
            fig_a4 = px.bar(
                w_df,
                x="Weekend_Status",
                y="Purchase_Rate_Pct",
                color="Weekend_Status",
                text="Purchase_Rate_Pct",
                hover_data=["Sessions", "Purchases"],
                color_discrete_map={"Weekday (False)": "#2F855A", "Weekend (True)": "#153E2D"},
                labels={"Weekend_Status": "Weekend Status", "Purchase_Rate_Pct": "Purchase Rate (%)"},
            )
            fig_a4.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            ui.apply_plotly_theme(fig_a4, height=310)
            fig_a4.update_layout(showlegend=False)
            st.plotly_chart(fig_a4, use_container_width=True)

    # -------------------------------------------------------------------------
    # B. BROWSING BEHAVIOR
    # -------------------------------------------------------------------------
    with eda_b:
        b1, b2 = st.columns(2)
        with b1, ui.chart_card("B1. Administrative, Informational & Product-Related Page Visits", "Mean page visits per session grouped by purchase outcome"):
            pg_grp = (
                df_eda.groupby("Outcome", as_index=False)[["Administrative", "Informational", "ProductRelated"]]
                .mean()
                .round(2)
            )
            pg_long = pg_grp.melt("Outcome", var_name="Page Type", value_name="Average Pages Visited")
            fig_b1 = px.bar(
                pg_long,
                x="Page Type",
                y="Average Pages Visited",
                color="Outcome",
                barmode="group",
                text="Average Pages Visited",
                color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
            )
            fig_b1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            ui.apply_plotly_theme(fig_b1, height=320)
            st.plotly_chart(fig_b1, use_container_width=True)

        with b2, ui.chart_card("B2. Browsing Durations Across Page Categories (Seconds)", "Mean duration spent on Administrative, Informational, and Product-Related pages"):
            dur_grp = (
                df_eda.groupby("Outcome", as_index=False)[
                    ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]
                ]
                .mean()
                .round(1)
            )
            dur_long = dur_grp.melt("Outcome", var_name="Duration Type", value_name="Mean Duration (sec)")
            fig_b2 = px.bar(
                dur_long,
                x="Duration Type",
                y="Mean Duration (sec)",
                color="Outcome",
                barmode="group",
                text="Mean Duration (sec)",
                color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
            )
            fig_b2.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
            ui.apply_plotly_theme(fig_b2, height=320)
            st.plotly_chart(fig_b2, use_container_width=True)

        b3, b4 = st.columns(2)
        with b3, ui.chart_card("B3. Distribution of Product-Related Page Visits", "Box plot comparing ProductRelated page counts (clipped at 99th percentile for readability)"):
            cap_p = float(df_eda["ProductRelated"].quantile(0.99))
            fig_b3 = px.box(
                df_eda[df_eda["ProductRelated"] <= cap_p],
                x="Outcome",
                y="ProductRelated",
                color="Outcome",
                color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
            )
            ui.apply_plotly_theme(fig_b3, height=310)
            fig_b3.update_layout(showlegend=False)
            st.plotly_chart(fig_b3, use_container_width=True)

        with b4, ui.chart_card("B4. Distribution of Administrative & Informational Visits", "Comparison of administrative and informational browsing depth"):
            adm_inf = df_eda.melt(
                id_vars=["Outcome"],
                value_vars=["Administrative", "Informational"],
                var_name="Page Category",
                value_name="Page Count",
            )
            fig_b4 = px.box(
                adm_inf,
                x="Page Category",
                y="Page Count",
                color="Outcome",
                color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
            )
            ui.apply_plotly_theme(fig_b4, height=310)
            st.plotly_chart(fig_b4, use_container_width=True)

    # -------------------------------------------------------------------------
    # C. WEBSITE INTERACTION
    # -------------------------------------------------------------------------
    with eda_c:
        c1, c2 = st.columns(2)
        with c1, ui.chart_card("C1. BounceRates & ExitRates by Purchase Outcome", "Purchase sessions exhibit lower average bounce rates (0.51% vs 2.53%) and exit rates (1.96% vs 4.74%)"):
            be_grp = (
                df_eda.groupby("Outcome", as_index=False)[["BounceRates", "ExitRates"]]
                .mean()
                .round(4)
            )
            be_long = be_grp.melt("Outcome", var_name="Rate Metric", value_name="Average Rate")
            fig_c1 = px.bar(
                be_long,
                x="Rate Metric",
                y="Average Rate",
                color="Outcome",
                barmode="group",
                text="Average Rate",
                color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
            )
            fig_c1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            ui.apply_plotly_theme(fig_c1, height=320)
            st.plotly_chart(fig_c1, use_container_width=True)

        with c2, ui.chart_card("C2. PageValues Distribution by Purchase Outcome", "80.6% of purchase sessions have PageValues > 0 (mean 27.26) vs 11.4% of non-purchase sessions (mean 1.98)"):
            pv_cap = float(df_eda["PageValues"].quantile(0.98))
            fig_c2 = px.box(
                df_eda[df_eda["PageValues"] <= pv_cap],
                x="Outcome",
                y="PageValues",
                color="Outcome",
                color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
            )
            ui.apply_plotly_theme(fig_c2, height=320)
            fig_c2.update_layout(showlegend=False)
            st.plotly_chart(fig_c2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3, ui.chart_card("C3. SpecialDay Closeness vs. Purchase Rate", "Session count and conversion rate by SpecialDay proximity index (0.0 to 1.0)"):
            sp_df = (
                df_eda.assign(SpecialDay_Str=df_eda["SpecialDay"].round(1).astype(str))
                .groupby("SpecialDay_Str", as_index=False)
                .agg(Sessions=("rev_int", "size"), Purchases=("rev_int", "sum"), Purchase_Rate=("rev_int", "mean"))
            )
            sp_df["Purchase_Rate_Pct"] = (sp_df["Purchase_Rate"] * 100).round(2)
            fig_c3 = px.bar(
                sp_df,
                x="SpecialDay_Str",
                y="Purchase_Rate_Pct",
                text="Purchase_Rate_Pct",
                hover_data=["Sessions", "Purchases"],
                color="Purchase_Rate_Pct",
                color_continuous_scale=["#A7C957", "#2F855A", "#153E2D"],
                labels={"SpecialDay_Str": "SpecialDay Closeness Index", "Purchase_Rate_Pct": "Purchase Rate (%)"},
            )
            fig_c3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            ui.apply_plotly_theme(fig_c3, height=310)
            fig_c3.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_c3, use_container_width=True)

        with c4, ui.chart_card("C4. ExitRates Histogram by Purchase Outcome", "Distribution of session exit rates across purchase and non-purchase sessions"):
            fig_c4 = px.histogram(
                df_eda,
                x="ExitRates",
                color="Outcome",
                nbins=40,
                barmode="overlay",
                opacity=0.72,
                color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
            )
            ui.apply_plotly_theme(fig_c4, height=310)
            st.plotly_chart(fig_c4, use_container_width=True)

    # -------------------------------------------------------------------------
    # D. FEATURE RELATIONSHIPS
    # -------------------------------------------------------------------------
    with eda_d:
        corr_cols = data_validation.CONTINUOUS_FEATURES + ["rev_int"]
        corr_mat = df_eda[corr_cols].rename(columns={"rev_int": "Revenue (0/1)"}).corr().round(3)

        with ui.chart_card("D1. Correlation Heatmap (Continuous Features & Binary Revenue)", "Pearson correlation matrix showing page count/duration pairs, Bounce-Exit correlation (0.913), and PageValues-Revenue correlation (0.493)"):
            fig_corr = px.imshow(
                corr_mat,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Greens",
                zmin=-0.3,
                zmax=1.0,
            )
            ui.apply_plotly_theme(fig_corr, height=440)
            st.plotly_chart(fig_corr, use_container_width=True)

        d1, d2 = st.columns(2)
        with d1, ui.chart_card("D2. Feature Distribution Explorer by Purchase Outcome", "Select any continuous feature to compare its distribution across Revenue outcomes"):
            chosen_feat = st.selectbox("Select Numerical Feature", options=data_validation.CONTINUOUS_FEATURES, index=4)
            fig_d2 = px.histogram(
                df_eda,
                x=chosen_feat,
                color="Outcome",
                nbins=45,
                barmode="overlay",
                opacity=0.72,
                marginal="box",
                color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
            )
            ui.apply_plotly_theme(fig_d2, height=340)
            st.plotly_chart(fig_d2, use_container_width=True)

        with d2, ui.chart_card("D3. Bivariate Relationship Explorer", "Inspect the joint relationship between any two numerical features (2,500-session stratified sample)"):
            sc1, sc2 = st.columns(2)
            x_feat = sc1.selectbox("X-Axis Feature", options=data_validation.CONTINUOUS_FEATURES, index=4, key="rel_x")
            y_feat = sc2.selectbox("Y-Axis Feature", options=data_validation.CONTINUOUS_FEATURES, index=5, key="rel_y")
            sample_rel = df_eda.sample(n=min(2500, len(df_eda)), random_state=42)
            fig_d3 = px.scatter(
                sample_rel,
                x=x_feat,
                y=y_feat,
                color="Outcome",
                opacity=0.65,
                hover_data=["Month", "VisitorType", "PageValues"],
                color_discrete_map={"No Purchase (False)": "#607D8B", "Purchase (True)": "#2F855A"},
            )
            ui.apply_plotly_theme(fig_d3, height=285)
            st.plotly_chart(fig_d3, use_container_width=True)
