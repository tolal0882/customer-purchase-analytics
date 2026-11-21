"""
Purchase Prediction — Interactive early-session purchase inference using the
compiled React + TypeScript + Tailwind PredictionForm component (with native
Streamlit fallback), gradient result card, and local feature-perturbation explanation.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
try:
    from components import react_bridge, ui
except ImportError:
    from frontend.components import react_bridge, ui

ui.inject_css()

try:
    schema = api_client.get_schema()
    report = api_client.get_metrics()
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

ui.page_header(
    "Purchase Behavior Prediction",
    "Estimate early-session purchase conversion likelihood from the first 3 non-transaction interactions using a saved classification pipeline.",
    badges=[
        (f"Active Model: {report.get('best_model', 'Logistic Regression')}", "ok"),
        (f"Threshold: {report.get('training_config', {}).get('threshold', 0.5):.2f}", "info"),
        ("Snapshot Policy: First 3 Non-Tx Events", "neutral"),
    ],
)

if not schema["model_trained"]:
    ui.empty_state(
        "No trained model available",
        "Train and save a classification pipeline on the **Model Training & Eval** page first.",
        "🛒",
    )
    st.stop()

# =============================================================================
# 1. ACTIVE SAVED PIPELINE SELECTOR & METRICS SUMMARY CARD
# =============================================================================
exps = report.get("experiments", [])
c_mod, c_info = st.columns([1.5, 2.5])
with c_mod:
    if len(exps) > 1:
        exp_options = {
            e["experiment_id"]: f"{e['best_model']} ({e['experiment_id']})"
            for e in exps
        }
        cur_id = report.get("experiment_id")
        chosen_exp = st.selectbox(
            "Saved Model Pipeline",
            options=list(exp_options.keys()),
            index=list(exp_options.keys()).index(cur_id) if cur_id in exp_options else 0,
            format_func=lambda k: exp_options[k],
        )
        if chosen_exp != cur_id:
            api_client.activate_experiment(chosen_exp)
            st.rerun()
    else:
        st.selectbox("Saved Model Pipeline", [f"{report['best_model']} (active)"], disabled=True)

with c_info:
    ui.model_summary_card(report)

ui.note(
    "<b>Model-based statistical estimate:</b> Predictions are derived from early-session behavioral signals "
    "and logged to Prediction History (<code>data/predictions.db</code>) separately from real dataset transactions."
)

# =============================================================================
# 2. EARLY-SESSION INPUT FORM (REACT + TAILWIND COMPONENT WITH NATIVE FALLBACK)
# =============================================================================
DEFAULT_VALUES = {
    "views_count": 2,
    "unique_items_viewed": 2,
    "add_to_cart_count": 1,
    "first_cart_position": 2,
    "elapsed_seconds": 90.0,
    "ends_with_cart": "Yes",
    "start_hour": 14,
    "start_dayofweek": 2,
    "is_weekend": "No",
    "prior_sessions": 1,
    "prior_purchases": 0,
}

if "pred_input_values" not in st.session_state:
    st.session_state["pred_input_values"] = dict(DEFAULT_VALUES)

left, right = st.columns([1.25, 1])
triggered = False
current_values = dict(st.session_state["pred_input_values"])
active_cols = schema.get("active_input_columns") or [f["name"] for f in schema["features"]]

with left:
    st.markdown('<div class="card-title">Early-Session Behaviour Inputs</div>', unsafe_allow_html=True)
    if react_bridge.is_available():
        out = react_bridge.prediction_form(
            features=active_cols,
            values=current_values,
            policy="n_events",
            n_events=3,
            reset_token="v1",
            key="pred_react_form",
        )
        if out is not None and isinstance(out, dict):
            nonce = out.get("nonce")
            if nonce and nonce != st.session_state.get("_last_pred_nonce"):
                st.session_state["_last_pred_nonce"] = nonce
                mapped = react_bridge.map_react_prediction_values(out.get("values", {}))
                current_values.update(mapped)
                st.session_state["pred_input_values"] = current_values
                triggered = True
    else:
        with st.form("predict_native_form", border=True):
            cols = st.columns(2)
            for i, field in enumerate(schema["features"]):
                box = cols[i % 2]
                fname = field["name"]
                label = field.get("label") or fname
                default_val = current_values.get(fname, field["default"])
                if field["type"] == "category":
                    opts = field["options"]
                    idx = opts.index(str(default_val)) if str(default_val) in opts else 0
                    current_values[fname] = box.selectbox(label, opts, index=idx)
                elif field["type"] == "integer":
                    min_v = -1 if fname == "first_cart_position" else 0
                    current_values[fname] = int(box.number_input(label, min_value=min_v, value=int(default_val), step=1))
                else:
                    current_values[fname] = float(box.number_input(label, min_value=0.0, value=float(default_val), step=5.0))
            cols[len(schema["features"]) % 2].number_input(
                "Cart Removals (unavailable in Retailrocket)", min_value=0, max_value=0, value=0, disabled=True
            )
            if st.form_submit_button("Predict purchase likelihood", type="primary", use_container_width=True):
                st.session_state["pred_input_values"] = current_values
                triggered = True

if triggered:
    try:
        payload = {k: current_values[k] for k in active_cols if k in current_values}
        st.session_state["last_pred_result"] = api_client.predict(payload)
        st.session_state["last_pred_input"] = payload
    except api_client.APIError as e:
        st.error(str(e))

# =============================================================================
# 3. PREDICTION RESULT HERO BANNER & LOCAL EXPLANATION CHART
# =============================================================================
with right:
    st.markdown('<div class="card-title">Prediction Output & Local Explanation</div>', unsafe_allow_html=True)
    res = st.session_state.get("last_pred_result")
    if res is None:
        # Auto-compute initial prediction on first visit so the system screen is live immediately
        try:
            payload = {k: current_values[k] for k in active_cols if k in current_values}
            res = api_client.predict(payload)
            st.session_state["last_pred_result"] = res
            st.session_state["last_pred_input"] = payload
        except Exception:
            res = None

    if res is None:
        ui.empty_state(
            "Ready to predict",
            "Choose an example preset or adjust session inputs on the left and click **Predict purchase likelihood**.",
            "🎯",
        )
    else:
        ui.prediction_result_card(res)
        if res.get("business_recommendation"):
            ui.note(f"<b>Recommended Operational Action:</b> {ui.esc(res['business_recommendation'])}")

        exp_list = res.get("explanation", [])
        if exp_list:
            with ui.chart_card(
                "What Drove This Estimate? (Feature Perturbation vs Training Baseline)",
                "Shift in predicted purchase probability relative to each feature's training-window median/mode",
            ):
                exp_df = pd.DataFrame(exp_list).head(8).sort_values("probability_effect", ascending=True)
                fig_exp = px.bar(
                    exp_df, x="probability_effect", y="feature", orientation="h",
                    color="probability_effect", color_continuous_scale="RdYlGn",
                    labels={"probability_effect": "Probability Shift", "feature": "Feature"},
                )
                fig_exp.update_layout(height=250, coloraxis_showscale=False, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_exp, use_container_width=True)

        with st.expander("Exact Feature Vector Passed to Saved Pipeline"):
            st.json(st.session_state.get("last_pred_input", current_values))
