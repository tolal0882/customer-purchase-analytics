"""
Page 3 – Purchase Prediction: schema-driven customer behavior input form
with example presets, calling POST /api/predict and displaying purchase
probability and actionable business recommendations.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client

st.title("🔮 Purchase Behavior Prediction")
st.caption(
    "Predict whether an active shopping session will convert into a purchase based on the customer's "
    "first 3 interactions and prior session history (Supervised Classification)."
)

try:
    schema = api_client.get_schema()
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

if not schema["model_trained"]:
    st.warning("The classification model is not trained yet. Go to **5. Model Performance** and click **Train Model**.")
    st.stop()

# =============================================================================
# 1. QUICK CUSTOMER SCENARIO PRESETS
# =============================================================================
PRESETS = {
    "Custom / Default": None,
    "High-Intent Cart Shopper": {
        "views_count": 2, "unique_items_viewed": 2, "add_to_cart_count": 1,
        "elapsed_seconds": 85.0, "ends_with_cart": "Yes", "start_hour": 14,
        "is_weekend": "No", "prior_sessions": 2, "prior_purchases": 1,
    },
    "Returning Product Researcher": {
        "views_count": 3, "unique_items_viewed": 1, "add_to_cart_count": 0,
        "elapsed_seconds": 140.0, "ends_with_cart": "No", "start_hour": 20,
        "is_weekend": "Yes", "prior_sessions": 4, "prior_purchases": 0,
    },
    "First-Time Casual Browser": {
        "views_count": 3, "unique_items_viewed": 3, "add_to_cart_count": 0,
        "elapsed_seconds": 35.0, "ends_with_cart": "No", "start_hour": 11,
        "is_weekend": "No", "prior_sessions": 0, "prior_purchases": 0,
    },
}

preset_name = st.radio("Load an example customer scenario:", list(PRESETS.keys()), horizontal=True)
preset_vals = PRESETS[preset_name] or {}

# =============================================================================
# 2. SCHEMA-DRIVEN INPUT FORM (FROM GET /api/schema)
# =============================================================================
left, right = st.columns([1.25, 1])
with left:
    with st.form("predict_form", border=True):
        st.subheader("Early-Session Customer Inputs")
        cols = st.columns(2)
        values = {}
        for i, field in enumerate(schema["features"]):
            box = cols[i % 2]
            fname = field["name"]
            label = field.get("label") or fname.replace("_", " ").capitalize()
            help_txt = field.get("help") or None
            default_val = preset_vals.get(fname, field["default"])

            if field["type"] == "category":
                opts = field["options"]
                idx = opts.index(str(default_val)) if str(default_val) in opts else 0
                values[fname] = box.selectbox(label, opts, index=idx, help=help_txt)
            elif field["type"] == "integer":
                values[fname] = int(
                    box.number_input(label, min_value=0, value=int(default_val), step=1, help=help_txt)
                )
            else:
                values[fname] = float(
                    box.number_input(label, min_value=0.0, value=float(default_val), step=5.0, help=help_txt)
                )

        submitted = st.form_submit_button("🔮 Predict Purchase Likelihood", type="primary", use_container_width=True)

# =============================================================================
# 3. PREDICTION RESULT & BUSINESS RECOMMENDATION
# =============================================================================
with right:
    with st.container(border=True):
        st.subheader("Prediction Result")
        if submitted:
            try:
                st.session_state["last_pred_result"] = api_client.predict(values)
                st.session_state["last_pred_input"] = values
            except api_client.APIError as e:
                st.error(str(e))
                st.stop()

        res = st.session_state.get("last_pred_result")
        if not res:
            st.info("Choose a scenario or enter values on the left, then click **Predict Purchase Likelihood**.")
        else:
            is_buy = res["prediction"] == schema.get("positive_class", "Yes")
            probs = res.get("probabilities", {})
            p_yes = float(probs.get("Yes", res.get("probability", 0.0)))

            if is_buy:
                st.success(f"### Predicted Outcome: **Purchase Likely (`{res['prediction']}`)**")
            else:
                st.warning(f"### Predicted Outcome: **No Purchase Expected (`{res['prediction']}`)**")

            k1, k2 = st.columns(2)
            k1.metric("Purchase Probability", f"{p_yes * 100:.1f}%")
            k2.metric("Customer Intent Tier", res.get("intent_tier", "—"))

            if probs:
                pdf = pd.DataFrame([
                    {"Outcome": "Purchase (Yes)", "Probability (%)": round(probs.get("Yes", 0) * 100, 1)},
                    {"Outcome": "No Purchase (No)", "Probability (%)": round(probs.get("No", 0) * 100, 1)},
                ])
                fig = px.bar(
                    pdf, x="Probability (%)", y="Outcome", orientation="h", color="Outcome",
                    text="Probability (%)",
                    color_discrete_map={"Purchase (Yes)": "#2F855A", "No Purchase (No)": "#94A3B8"},
                )
                fig.update_layout(height=180, showlegend=False, xaxis_range=[0, 105], margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            if res.get("business_recommendation"):
                st.info(f"**Recommended Action:** {res['business_recommendation']}")
            st.caption("Saved automatically to SQLite prediction history (view on **6. Prediction History**).")
