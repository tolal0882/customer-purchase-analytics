"""
Page 3 – Purchase Prediction: interactive early-session behavior form
with consistency validation, model selector, predicted purchase class & probability,
and local feature-perturbation explanation chart.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client

st.title("🔮 Purchase Behavior Prediction")
st.caption(
    "Estimate whether an active shopping session will convert into a purchase based on early-session behavior "
    "(first 3 non-transaction interactions + historical customer context) using the saved classification pipeline."
)

try:
    schema = api_client.get_schema()
    report = api_client.get_metrics()
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

if not schema["model_trained"]:
    st.warning("No compatible trained model is available. Go to **5. Model Performance** and train a model first.")
    st.stop()

# =============================================================================
# 1. ACTIVE MODEL SELECTOR & DISCLAIMER
# =============================================================================
exps = report.get("experiments", [])
if len(exps) > 1:
    with st.container(border=True):
        c_sel, c_meta = st.columns([1.8, 2.2])
        exp_options = {
            e["experiment_id"]: f"{e['best_model']} ({e['experiment_id']}) · Test F1={e['test_metrics']['f1']:.3f} · ROC-AUC={e['test_metrics']['roc_auc']:.3f}"
            for e in exps
        }
        cur_id = report.get("experiment_id")
        chosen_exp = c_sel.selectbox(
            "Active Saved Model Pipeline",
            options=list(exp_options.keys()),
            index=list(exp_options.keys()).index(cur_id) if cur_id in exp_options else 0,
            format_func=lambda k: exp_options[k],
        )
        if chosen_exp != cur_id:
            api_client.activate_experiment(chosen_exp)
            st.rerun()
        c_meta.info(
            "**Disclaimer:** This is a **model-based statistical estimate** from early-session behavior, "
            "not a confirmed transaction. Predictions are logged separately in SQLite history."
        )
else:
    st.info(
        f"**Active Model:** `{report['best_model']}` (Trained `{report['trained_at']}`) · "
        "**Note:** This is a model-based statistical estimate, not a confirmed purchase."
    )

# =============================================================================
# 2. SCENARIO PRESETS & CONSISTENCY-VALIDATED INPUT FORM
# =============================================================================
PRESETS = {
    "Custom / Default": None,
    "High-Intent Cart Shopper": {
        "views_count": 2, "unique_items_viewed": 2, "add_to_cart_count": 1,
        "first_cart_position": 2, "elapsed_seconds": 85.0, "ends_with_cart": "Yes",
        "start_hour": 14, "start_dayofweek": 2, "is_weekend": "No",
        "prior_sessions": 2, "prior_purchases": 1,
    },
    "Focused Product Revisitor": {
        "views_count": 3, "unique_items_viewed": 1, "add_to_cart_count": 0,
        "first_cart_position": -1, "elapsed_seconds": 140.0, "ends_with_cart": "No",
        "start_hour": 20, "start_dayofweek": 5, "is_weekend": "Yes",
        "prior_sessions": 4, "prior_purchases": 0,
    },
    "First-Time Quick Browser": {
        "views_count": 3, "unique_items_viewed": 3, "add_to_cart_count": 0,
        "first_cart_position": -1, "elapsed_seconds": 30.0, "ends_with_cart": "No",
        "start_hour": 11, "start_dayofweek": 1, "is_weekend": "No",
        "prior_sessions": 0, "prior_purchases": 0,
    },
}

preset_name = st.radio("Quick Scenario Presets:", list(PRESETS.keys()), horizontal=True)
preset_vals = PRESETS[preset_name] or {}

left, right = st.columns([1.25, 1])
with left:
    with st.form("predict_form", border=True):
        st.subheader("Early-Session Behavior Inputs")
        cols = st.columns(2)
        values = {}
        active_fields = [f for f in schema["features"] if f.get("active_in_model", True)]
        for i, field in enumerate(active_fields):
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
                min_v = -1 if fname == "first_cart_position" else 0
                values[fname] = int(
                    box.number_input(label, min_value=min_v, value=int(default_val), step=1, help=help_txt)
                )
            else:
                values[fname] = float(
                    box.number_input(label, min_value=0.0, value=float(default_val), step=5.0, help=help_txt)
                )

        # Disabled informational field explaining why cart removals are omitted
        cols[len(active_fields) % 2].number_input(
            "Cart Removals (unavailable in Retailrocket)",
            min_value=0, max_value=0, value=0, disabled=True,
            help="Retailrocket logs view, addtocart, and transaction only; no removefromcart event exists.",
        )

        submitted = st.form_submit_button("🔮 Predict Purchase Outcome", type="primary", use_container_width=True)

# =============================================================================
# 3. VALIDATION, PREDICTION OUTPUT & LOCAL FEATURE EXPLANATION
# =============================================================================
with right:
    with st.container(border=True):
        st.subheader("Prediction Result & Explanation")
        if submitted:
            # Validate logical consistency of session inputs
            errs = []
            v_cnt = int(values.get("views_count", 2))
            c_cnt = int(values.get("add_to_cart_count", 0))
            u_cnt = int(values.get("unique_items_viewed", 1))
            if v_cnt + c_cnt == 0:
                errs.append("Total interactions (views + add-to-cart) must be at least 1.")
            if u_cnt > v_cnt + c_cnt:
                errs.append(f"Unique products viewed ({u_cnt}) cannot exceed total interactions ({v_cnt + c_cnt}).")
            if c_cnt == 0 and values.get("ends_with_cart") == "Yes":
                errs.append("'Latest Action Was Add-to-Cart' cannot be Yes when Add-to-Cart Actions is 0.")
            if c_cnt == 0 and int(values.get("first_cart_position", -1)) >= 0:
                errs.append("First Add-to-Cart Position must be -1 when Add-to-Cart Actions is 0.")

            if errs:
                for err in errs:
                    st.error(err)
            else:
                try:
                    st.session_state["last_pred_result"] = api_client.predict(values)
                    st.session_state["last_pred_input"] = values
                except api_client.APIError as e:
                    st.error(str(e))

        res = st.session_state.get("last_pred_result")
        if not res:
            st.info("Enter session behavior on the left and click **Predict Purchase Outcome**.")
        else:
            is_buy = res["prediction"] == schema.get("positive_class", "Yes")
            probs = res.get("probabilities", {})
            p_yes = float(probs.get("Yes", res.get("probability", 0.0)))
            thr = float(res.get("threshold", 0.5))

            if is_buy:
                st.success(f"### Predicted Class: **Purchase (`{res['prediction']}` — Class 1)**")
            else:
                st.warning(f"### Predicted Class: **No Purchase (`{res['prediction']}` — Class 0)**")

            k1, k2, k3 = st.columns(3)
            k1.metric("Purchase Probability", f"{p_yes * 100:.1f}%", f"Threshold: {thr * 100:.0f}%")
            k2.metric("Model Used", res.get("model_name", report["best_model"]))
            k3.metric("Intent Tier", res.get("intent_tier", "—"))

            if res.get("business_recommendation"):
                st.info(f"**Recommended Action:** {res['business_recommendation']}")

            exp_list = res.get("explanation", [])
            if exp_list:
                st.markdown("**Local Model Explanation (Feature Effect vs Training Baseline)**")
                exp_df = pd.DataFrame(exp_list).head(8).sort_values("probability_effect", ascending=True)
                fig_exp = px.bar(
                    exp_df, x="probability_effect", y="feature", orientation="h",
                    color="probability_effect", color_continuous_scale="RdYlGn",
                    labels={"probability_effect": "Probability Shift", "feature": "Feature"},
                )
                fig_exp.update_layout(height=250, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_exp, use_container_width=True)

            with st.expander("Input Summary Sent to Model Pipeline"):
                st.json(st.session_state.get("last_pred_input", values))
