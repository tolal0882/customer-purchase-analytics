"""
Prediction History — Persistent SQLite prediction log with search, filtering,
sorting, CSV export, single record inspector, and confirmation before clearing.
"""
import pandas as pd
import streamlit as st

import api_client
try:
    from components import ui
except ImportError:
    from frontend.components import ui

ui.inject_css()

ui.page_header(
    "Prediction History Log",
    "Persistent SQLite log (data/predictions.db) of early-session purchase predictions, kept separate from real dataset transactions.",
    badges=[
        ("Storage: SQLite (data/predictions.db)", "info"),
        ("CSV Export Enabled", "neutral"),
    ],
)

try:
    items = api_client.get_history(limit=250)["items"]
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

if not items:
    ui.empty_state(
        "No prediction records logged yet",
        "Run a prediction on the **Purchase Prediction** page to populate the history table.",
        "📜",
    )
    st.stop()

rows = [
    {
        "prediction_id": it.get("id"),
        "timestamp": it["created_at"],
        "model": it.get("model_name", "Classifier"),
        "predicted_outcome": it["prediction"],
        "purchase_probability_pct": round(float(it.get("probability") or 0.0) * 100, 2),
        "intent_tier": it.get("intent_tier", "—"),
        **it["input"],
    }
    for it in items
]
df = pd.DataFrame(rows)

ui.kpi_row([
    ("Total Logged Predictions", f"{len(df):,}", "Persisted in SQLite"),
    ("Predicted Purchases (Yes)", f"{int((df['predicted_outcome'] == 'Yes').sum()):,}", "Class 1 predictions"),
    ("Mean Purchase Probability", f"{df['purchase_probability_pct'].mean():.1f}%", "Across logged inferences"),
    ("Distinct Models Used", f"{df['model'].nunique()}", "Active & historical pipelines"),
])

with ui.chart_card("Search, Filter & Sort Prediction Log"):
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.2])
    search_q = c1.text_input("Search ID, model, or input features")
    out_filter = c2.multiselect(
        "Predicted Outcome",
        options=sorted(df["predicted_outcome"].unique()),
        default=sorted(df["predicted_outcome"].unique()),
    )
    mod_filter = c3.multiselect(
        "Model",
        options=sorted(df["model"].unique()),
        default=sorted(df["model"].unique()),
    )
    sort_by = c4.selectbox(
        "Sort Order",
        ["Newest First", "Oldest First", "Probability (High → Low)", "Probability (Low → High)"],
    )

view = df[df["predicted_outcome"].isin(out_filter) & df["model"].isin(mod_filter)].copy()
if search_q.strip():
    mask = view.astype(str).apply(lambda s: s.str.contains(search_q.strip(), case=False, na=False, regex=False)).any(axis=1)
    view = view[mask]

if sort_by == "Newest First":
    view = view.sort_values("prediction_id", ascending=False)
elif sort_by == "Oldest First":
    view = view.sort_values("prediction_id", ascending=True)
elif sort_by == "Probability (High → Low)":
    view = view.sort_values("purchase_probability_pct", ascending=False)
else:
    view = view.sort_values("purchase_probability_pct", ascending=True)

b1, b2 = st.columns(2)
b1.download_button(
    f"⬇️ Export Filtered Prediction History ({len(view):,} rows CSV)",
    view.to_csv(index=False).encode("utf-8"),
    file_name="prediction_history.csv",
    mime="text/csv",
    use_container_width=True,
)
if b2.button("🗑️ Clear Prediction History", use_container_width=True):
    st.session_state["confirm_clear_history"] = True

if st.session_state.get("confirm_clear_history"):
    with st.container(border=True):
        st.warning(f"Confirm permanent deletion of all **{len(df):,}** logged prediction records?")
        y_col, n_col = st.columns(2)
        if y_col.button("✅ Yes, Delete All Records", type="primary", use_container_width=True):
            api_client.clear_history()
            st.session_state["confirm_clear_history"] = False
            st.rerun()
        if n_col.button("Cancel", use_container_width=True):
            st.session_state["confirm_clear_history"] = False
            st.rerun()

with ui.chart_card(f"Logged Prediction Records ({len(view):,} matching rows)"):
    st.dataframe(view, use_container_width=True, hide_index=True)

if not view.empty:
    with st.expander("🔎 Inspect Single Prediction Record Payload"):
        sel_id = st.selectbox("Select Prediction ID", view["prediction_id"].tolist())
        rec = next((it for it in items if it.get("id") == sel_id), None)
        if rec:
            st.json(rec)
