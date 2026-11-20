"""
Page 6 – Prediction History: review previous prediction records persisted in SQLite,
filter by predicted class or intent tier, export to CSV, or clear history.
"""
import pandas as pd
import streamlit as st

import api_client

st.title("🕘 Prediction History")
st.caption("Persistent log of purchase predictions stored in SQLite (`data/predictions.db`) via the FastAPI backend.")

try:
    items = api_client.get_history(limit=200)["items"]
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

if not items:
    st.info("No prediction records yet. Go to **3. Purchase Prediction** and run a prediction.")
    st.stop()

# =============================================================================
# 1. BUILD HISTORY DATAFRAME & SUMMARY KPIs
# =============================================================================
rows = [
    {
        "id": it.get("id"),
        "timestamp": it["created_at"],
        "prediction": it["prediction"],
        "purchase_probability_pct": round(float(it.get("probability") or 0.0) * 100, 1),
        "intent_tier": it.get("intent_tier", "—"),
        **it["input"],
    }
    for it in items
]
df = pd.DataFrame(rows)

k1, k2, k3 = st.columns(3)
k1.metric("Total Logged Predictions", f"{len(df):,}")
k2.metric("Predicted Purchases (Yes)", f"{int((df['prediction'] == 'Yes').sum()):,}")
k3.metric("Avg Purchase Probability", f"{df['purchase_probability_pct'].mean():.1f}%")

# =============================================================================
# 2. FILTER, EXPORT & CLEAR CONTROLS
# =============================================================================
c1, c2, c3 = st.columns([1.5, 1, 1])
pred_filter = c1.multiselect(
    "Filter by Predicted Outcome",
    options=sorted(df["prediction"].unique()),
    default=sorted(df["prediction"].unique()),
)
view = df[df["prediction"].isin(pred_filter)] if pred_filter else df

c2.download_button(
    "⬇️ Export History (CSV)",
    view.to_csv(index=False).encode("utf-8"),
    file_name="prediction_history.csv",
    mime="text/csv",
    use_container_width=True,
)
if c3.button("🗑️ Clear History", use_container_width=True):
    api_client.clear_history()
    st.success("Prediction history cleared.")
    st.rerun()

with st.container(border=True):
    st.dataframe(view, use_container_width=True, hide_index=True)
