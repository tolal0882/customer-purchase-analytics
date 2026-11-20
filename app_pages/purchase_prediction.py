"""Page 8: Purchase Prediction (interactive inference using the saved classification pipeline)."""
import pandas as pd
import streamlit as st

from app_pages import common
from components.streamlit_components import react_components as rc, ui
from src import charts, model_manager, prediction

DEFAULT_INPUTS = {
    "total_events": 3,
    "total_views": 2,
    "unique_items_viewed": 2,
    "add_to_cart_count": 1,
    "session_duration_sec": 90.0,
    "first_cart_position": 2,
    "ends_with_cart": True,
    "start_hour": 14,
    "start_dayofweek": 2,
    "hist_prior_sessions": 0,
    "hist_prior_purchases": 0,
    "hist_hours_since_prev_session": 0.0,
}


def _native_form(features: list[str], policy: str, n_events: int, vals: dict) -> tuple[bool, dict]:
    fset = set(features)
    with st.form("pred_native_form"):
        c1, c2, c3 = st.columns(3)
        v = dict(vals)
        v["total_events"] = int(c1.number_input("Events so far", 1, 500, int(v["total_events"]),
                                                help=f"Training snapshot used {n_events}" if policy == "n_events" else ""))
        v["total_views"] = int(c2.number_input("Product views", 0, 500, int(v["total_views"])))
        v["unique_items_viewed"] = int(c3.number_input("Unique products viewed", 0, 500, int(v["unique_items_viewed"])))
        c4, c5, c6 = st.columns(3)
        v["add_to_cart_count"] = int(c4.number_input("Add-to-cart count", 0, 500, int(v["add_to_cart_count"])))
        c5.number_input("Cart removals (unavailable in dataset)", 0, 0, 0, disabled=True)
        v["session_duration_sec"] = float(c6.number_input("Session elapsed time (sec)", 0.0, 86400.0, float(v["session_duration_sec"]), 5.0))
        c7, c8, c9 = st.columns(3)
        v["first_cart_position"] = int(c7.number_input("First add-to-cart 0-based position (-1 if none)", -1, 500, int(v["first_cart_position"])))
        v["ends_with_cart"] = bool(c8.checkbox("Last event was add-to-cart", value=bool(v["ends_with_cart"])))
        v["start_hour"] = int(c9.number_input("Start hour (0–23 UTC)", 0, 23, int(v["start_hour"])))
        v["start_dayofweek"] = int(st.selectbox("Weekday (0=Mon .. 6=Sun)", list(range(7)), index=int(v["start_dayofweek"])))
        if "hist_prior_sessions" in fset:
            h1, h2, h3 = st.columns(3)
            v["hist_prior_sessions"] = int(h1.number_input("Visitor prior sessions", 0, 1000, int(v["hist_prior_sessions"])))
            v["hist_prior_purchases"] = int(h2.number_input("Visitor prior purchases", 0, 1000, int(v["hist_prior_purchases"])))
            v["hist_hours_since_prev_session"] = float(h3.number_input("Hours since previous session", 0.0, 10000.0, float(v["hist_hours_since_prev_session"] or 0.0)))
        sub = st.form_submit_button("Predict purchase likelihood", type="primary", use_container_width=True)
    return sub, v


def render():
    ui.page_header(
        "Purchase Prediction",
        "Estimate early-session purchase probability using a saved, compatible Scikit-learn classification pipeline.",
    )
    exps = model_manager.list_experiments()
    if not exps:
        ui.empty_state(
            "No trained model available",
            "Predictions require a saved classification pipeline. Train and save a model on the **Model Training** page first.",
            "🛒",
        )
        return

    active_id = model_manager.get_active_experiment_id() or exps[0]["experiment_id"]
    all_ids = [e["experiment_id"] for e in exps]
    c_mod, c_info = st.columns([1.6, 2.4])
    chosen_id = c_mod.selectbox(
        "Saved model pipeline",
        all_ids,
        index=all_ids.index(active_id) if active_id in all_ids else 0,
        format_func=lambda i: f"{next(e['model_name'] for e in exps if e['experiment_id'] == i)} ({i})",
        key="pred_exp_select",
    )
    if chosen_id != active_id:
        model_manager.set_active_experiment(chosen_id)
    meta = model_manager.load_experiment(chosen_id)
    try:
        pipe = common.get_pipeline(chosen_id)
    except Exception as exc:
        ui.empty_state("Saved pipeline could not be loaded", f"Error loading `{meta.get('pipeline_file')}`: {exc}", "⚠️")
        return

    sp = meta.get("snapshot_params", {})
    with c_info:
        ui.model_comparison_card(meta)
        st.caption(
            f"Snapshot policy: **{sp.get('policy')}** "
            + (f"({sp.get('n_events')} non-transaction events)" if sp.get("policy") == "n_events" else f"({sp.get('cutoff_minutes')} min)")
            + f" · Threshold: **{meta['threshold']:.2f}** · Dataset version: `{meta['dataset_version']}`"
        )

    ui.note(
        "<b>Important disclaimer:</b> Predictions are <b>model-based statistical estimates</b> derived from early-session behaviour, "
        "not confirmed transactions. They are logged separately in Prediction History and never mixed with real dataset transactions."
    )

    if "pred_values" not in st.session_state:
        st.session_state["pred_values"] = dict(DEFAULT_INPUTS)
    if "pred_reset_token" not in st.session_state:
        st.session_state["pred_reset_token"] = "init"

    left, right = st.columns([1.25, 1])
    triggered = False
    raw_values = st.session_state["pred_values"]

    with left:
        st.markdown('<div class="card-title">Early-session behaviour inputs</div>', unsafe_allow_html=True)
        out = rc.prediction_form(
            features=meta["features"],
            policy=sp.get("policy", "n_events"),
            n_events=int(sp.get("n_events", 3)),
            values=raw_values,
            reset_token=st.session_state["pred_reset_token"],
            key="pred_react_form",
        )
        if out is not None and isinstance(out, dict):
            nonce = out.get("nonce")
            if nonce and nonce != st.session_state.get("_last_pred_nonce"):
                st.session_state["_last_pred_nonce"] = nonce
                raw_values = out.get("values", raw_values)
                st.session_state["pred_values"] = raw_values
                triggered = True
        elif not rc.react_available():
            st.caption("Using native Streamlit controls (React build unavailable).")
            triggered, raw_values = _native_form(
                meta["features"], sp.get("policy", "n_events"), int(sp.get("n_events", 3)), raw_values
            )
            st.session_state["pred_values"] = raw_values

    if triggered:
        feats, warns, errs = prediction.derive_features(raw_values, meta)
        st.session_state["pred_warns"] = warns
        st.session_state["pred_errs"] = errs
        if not errs:
            res = prediction.predict_one(pipe, meta, feats)
            pid = prediction.append_history(res, meta)
            res["prediction_id"] = pid
            st.session_state["pred_last_result"] = res

    with right:
        st.markdown('<div class="card-title">Prediction output & local explanation</div>', unsafe_allow_html=True)
        for e in st.session_state.get("pred_errs", []):
            st.error(e)
        for w in st.session_state.get("pred_warns", []):
            st.warning(w)
        res = st.session_state.get("pred_last_result")
        if res is None:
            ui.empty_state(
                "Ready to predict",
                "Enter session behaviour on the left (or pick an example preset) and click **Predict purchase likelihood**.",
                "🎯",
            )
        else:
            ui.prediction_result_card(res)
            st.caption(f"Logged to Prediction History with ID `{res.get('prediction_id', '—')}`.")
            with ui.chart_card("What drove this estimate? (Feature perturbation vs training median)",
                               "Change in predicted purchase probability when each feature is replaced by its training-window median"):
                exp_rows = res.get("explanation", [])
                if exp_rows:
                    imp_map = {r["feature"]: r["probability_effect"] for r in exp_rows}
                    st.plotly_chart(charts.importance_fig(imp_map, signed=True, height=280), use_container_width=True)
                    st.dataframe(pd.DataFrame(exp_rows).round(4), use_container_width=True, hide_index=True)
            with st.expander("Exact feature vector passed to the saved pipeline"):
                st.json(res["inputs"])
