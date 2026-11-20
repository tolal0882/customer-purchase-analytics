"""Page 9: Prediction History (local log of model predictions, kept strictly separate from dataset transactions)."""
import json

import pandas as pd
import streamlit as st

from app_pages import common
from components.streamlit_components import ui
from src import prediction


def render():
    ui.page_header(
        "Prediction History",
        "Inspect, filter, sort, and export saved model predictions. Prediction records are stored locally in data/history/ and never alter actual transaction data.",
    )
    df = prediction.read_history()
    if df.empty:
        ui.empty_state(
            "No prediction records yet",
            "Run a prediction on the **Purchase Prediction** page to populate local prediction history.",
            "🕘",
        )
        return

    pos_cnt = int((df["predicted_outcome"] == "Purchase").sum())
    ui.kpi_row([
        ("Total logged predictions", common.fmt_int(len(df)), "data/history/prediction_history.csv"),
        ("Predicted purchases", common.fmt_int(pos_cnt), f"{pos_cnt / len(df) * 100:.1f}% of runs"),
        ("Mean predicted probability", f"{df['purchase_probability'].mean() * 100:.1f}%", f"max {df['purchase_probability'].max() * 100:.1f}%"),
        ("Distinct models used", common.fmt_int(df["model_version"].nunique()), ""),
    ])

    with ui.chart_card("Search, filter & sort"):
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
        q = c1.text_input("Search ID, model, or input summary", key="ph_q")
        out_f = c2.multiselect("Predicted outcome", sorted(df["predicted_outcome"].unique()),
                               default=sorted(df["predicted_outcome"].unique()), key="ph_out")
        mod_f = c3.multiselect("Model", sorted(df["model"].unique()),
                               default=sorted(df["model"].unique()), key="ph_mod")
        sort_by = c4.selectbox("Sort by", ["timestamp (newest)", "timestamp (oldest)", "probability (high→low)", "probability (low→high)"], key="ph_sort")

    view = df[df["predicted_outcome"].isin(out_f) & df["model"].isin(mod_f)].copy()
    if q.strip():
        mask = (
            view["prediction_id"].astype(str).str.contains(q, case=False, na=False, regex=False)
            | view["model_version"].astype(str).str.contains(q, case=False, na=False, regex=False)
            | view["input_summary"].astype(str).str.contains(q, case=False, na=False, regex=False)
        )
        view = view[mask]
    if sort_by == "timestamp (newest)":
        view = view.sort_values("timestamp", ascending=False, kind="stable")
    elif sort_by == "timestamp (oldest)":
        view = view.sort_values("timestamp", ascending=True, kind="stable")
    elif sort_by == "probability (high→low)":
        view = view.sort_values("purchase_probability", ascending=False, kind="stable")
    else:
        view = view.sort_values("purchase_probability", ascending=True, kind="stable")

    b1, b2 = st.columns([1, 1])
    b1.download_button(
        f"⬇️ Export filtered history ({len(view):,} rows CSV)",
        view.to_csv(index=False).encode(),
        "prediction_history_export.csv",
        "text/csv",
        use_container_width=True,
    )
    if b2.button("🗑️ Clear all prediction history", use_container_width=True):
        st.session_state["_ask_clear_hist"] = True
    if st.session_state.get("_ask_clear_hist"):
        if ui.confirm_dialog("clear_hist", "Clear prediction history?",
                             f"Permanently delete all {len(df):,} prediction records from data/history/prediction_history.csv?",
                             "Clear history"):
            st.session_state.pop("_ask_clear_hist", None)
            prediction.clear_history()
            st.rerun()

    if view.empty:
        ui.empty_state("No matching prediction records", "Adjust or clear the search and filter criteria.", "🔍")
        return

    with ui.chart_card("Prediction log"):
        st.dataframe(view, use_container_width=True, hide_index=True)

    with ui.chart_card("Inspect single prediction record"):
        pid = st.selectbox("Select Prediction ID", view["prediction_id"].tolist(), key="ph_inspect")
        row = view[view["prediction_id"] == pid].iloc[0]
        c1, c2 = st.columns(2)
        c1.markdown(
            f"- **Prediction ID:** `{row['prediction_id']}`\n"
            f"- **Timestamp:** `{row['timestamp']}`\n"
            f"- **Model:** `{row['model']}` (`{row['model_version']}`)\n"
            f"- **Predicted outcome:** **{row['predicted_outcome']}**\n"
            f"- **Purchase probability:** **{float(row['purchase_probability']) * 100:.2f}%** (threshold {float(row['threshold']) * 100:.0f}%)"
        )
        try:
            parsed = json.loads(row["input_summary"])
        except Exception:
            parsed = {"raw": row["input_summary"]}
        c2.markdown("**Input feature vector**")
        c2.json(parsed)
