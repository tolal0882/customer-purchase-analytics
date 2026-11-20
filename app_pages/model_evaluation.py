"""Page 7: Model Evaluation (metrics, ROC/PR curves, confusion matrix, feature importance, precision-recall trade-off)."""
import pandas as pd
import streamlit as st

from app_pages import common
from components.streamlit_components import react_components as rc, ui
from src import charts, model_manager


def render():
    ui.page_header(
        "Model Evaluation",
        "Inspect held-out test metrics, confusion matrices, ROC and Precision–Recall curves, feature importances, and compare saved experiments.",
    )
    exps = model_manager.list_experiments()
    if not exps:
        ui.empty_state(
            "No trained models available",
            "Go to **Model Training** to train and save at least one classification pipeline. This page never displays fabricated or placeholder scores.",
            "📉",
        )
        return

    active_id = model_manager.get_active_experiment_id()
    all_ids = [e["experiment_id"] for e in exps]
    default_sel = st.session_state.get("eval_selected_ids") or all_ids[: min(4, len(all_ids))]
    default_sel = [i for i in default_sel if i in all_ids] or [all_ids[0]]
    rank_metric = st.session_state.get("eval_rank_metric", "pr_auc")

    with ui.chart_card("1 · Experiment leaderboard & active model selector",
                       "Select experiments to overlay on the ROC / PR curves and click 'Set active' to choose the pipeline used on Purchase Prediction."):
        react_payload = [
            {
                "id": e["experiment_id"],
                "label": f"{e['model_name']} ({e['experiment_id']})",
                "model": e["model_name"],
                "created_at": e["created_at"],
                "active": e["experiment_id"] == active_id,
                "metrics": {k: e["metrics"].get(k) for k in ("pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy")},
            }
            for e in exps
        ]
        out = rc.model_compare(
            react_payload,
            selected=default_sel,
            metric=rank_metric,
            metrics=["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"],
            key="eval_react_cmp",
        )
        if out is not None:
            if out.get("set_active") and out["set_active"] != active_id:
                model_manager.set_active_experiment(out["set_active"])
                st.rerun()
            if out.get("selected"):
                default_sel = [i for i in out["selected"] if i in all_ids] or default_sel
                st.session_state["eval_selected_ids"] = default_sel
            if out.get("metric"):
                rank_metric = out["metric"]
                st.session_state["eval_rank_metric"] = rank_metric
        else:
            c1, c2 = st.columns([2, 1])
            default_sel = c1.multiselect("Experiments to compare", all_ids, default=default_sel,
                                         format_func=lambda i: f"{i} ({model_manager.load_experiment(i)['model_name']})")
            new_act = c2.selectbox("Active model for Purchase Prediction", all_ids,
                                   index=all_ids.index(active_id) if active_id in all_ids else 0)
            if new_act != active_id:
                model_manager.set_active_experiment(new_act)
                st.rerun()

    sel_exps = [e for e in exps if e["experiment_id"] in default_sel] or [exps[0]]
    tbl = model_manager.experiments_table(sel_exps)
    tbl["label"] = tbl["model"] + " · " + tbl["experiment_id"].str[-8:]

    a, b = st.columns(2)
    with a, ui.chart_card("2 · Multi-metric comparison across selected experiments"):
        st.plotly_chart(
            charts.model_compare_fig(tbl, ["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"]),
            use_container_width=True,
        )
    with b, ui.chart_card("3 · Precision–Recall vs ROC curves (overlaid)"):
        roc_curves = {f"{e['model_name']} ({e['experiment_id'][-6:]})": e["metrics"]["roc_curve"]
                      for e in sel_exps if "roc_curve" in e["metrics"]}
        pr_curves = {f"{e['model_name']} ({e['experiment_id'][-6:]})": e["metrics"]["pr_curve"]
                     for e in sel_exps if "pr_curve" in e["metrics"]}
        curve_tab1, curve_tab2 = st.tabs(["Precision–Recall Curve", "ROC Curve"])
        with curve_tab1:
            prev = sel_exps[0]["metrics"].get("prevalence")
            st.plotly_chart(charts.pr_fig(pr_curves, prevalence=prev, height=310), use_container_width=True)
        with curve_tab2:
            st.plotly_chart(charts.roc_fig(roc_curves, height=310), use_container_width=True)

    ui.note(
        "<b>Precision / Recall trade-off on imbalanced purchase data:</b> Only ~7.5% of eligible sessions end in a purchase. "
        "Class weighting or lowering the decision threshold increases <b>Recall</b> (catching ~60–70% of real buyers) at the cost of lower "
        "<b>Precision</b> (~21–28%, meaning roughly 1 in 4 flagged sessions actually converts — still >3× the 7.5% random baseline). "
        "Raising the threshold increases Precision for high-cost interventions (e.g. discounts) while missing more buyers."
    )

    st.markdown("---")
    c_sel, c_act, c_del = st.columns([2.5, 1, 1])
    inspect_id = c_sel.selectbox(
        "4 · Deep-dive into a single experiment",
        all_ids,
        index=all_ids.index(active_id) if active_id in all_ids else 0,
        format_func=lambda i: f"{i} — {next(e['model_name'] for e in exps if e['experiment_id'] == i)}"
                              + (" [ACTIVE]" if i == active_id else ""),
        key="eval_inspect_id",
    )
    exp = model_manager.load_experiment(inspect_id)
    mt = exp["metrics"]
    if c_act.button("✅ Set as Active Model", use_container_width=True, disabled=(inspect_id == active_id)):
        model_manager.set_active_experiment(inspect_id)
        st.rerun()
    if c_del.button("🗑️ Delete Experiment", use_container_width=True):
        st.session_state["_ask_del_exp"] = inspect_id
    if st.session_state.get("_ask_del_exp") == inspect_id:
        if ui.confirm_dialog(f"del_{inspect_id}", "Delete saved experiment?",
                             f"Permanently delete experiment {inspect_id} and its joblib pipeline?", "Delete"):
            st.session_state.pop("_ask_del_exp", None)
            model_manager.delete_experiment(inspect_id)
            st.rerun()

    ui.kpi_row([
        ("Accuracy", f"{mt['accuracy']:.3f}", f"Majority baseline: {mt['majority_baseline_accuracy']:.3f}"),
        ("Precision", f"{mt['precision']:.3f}", f"Predicted positives: {common.fmt_int(mt['predicted_positive'])}"),
        ("Recall", f"{mt['recall']:.3f}", "True positives caught"),
        ("F1-score", f"{mt['f1']:.3f}", "Harmonic mean of P & R"),
        ("ROC-AUC", f"{mt['roc_auc']:.3f}", "Ranking across thresholds"),
        ("PR-AUC", f"{mt['pr_auc']:.3f}", f"Prevalence baseline: {mt['prevalence']:.3f}"),
        ("Train / Test N", f"{common.fmt_int(exp['split']['train_samples'])} / {common.fmt_int(exp['split']['test_samples'])}", "Chronological split"),
        ("Decision threshold", f"{exp['threshold']:.2f}", f"Model: {exp['model_name']}"),
    ])

    d1, d2 = st.columns(2)
    with d1, ui.chart_card("Confusion matrix (held-out chronological test window)",
                           "Rows = actual class, columns = predicted class"):
        st.plotly_chart(charts.confusion_fig(mt["confusion_matrix"]), use_container_width=True)
    with d2, ui.chart_card("Classification report"):
        rep_dict = mt["classification_report"]
        rows = []
        for k in ("No purchase", "Purchase", "macro avg", "weighted avg"):
            if k in rep_dict and isinstance(rep_dict[k], dict):
                rows.append({"Class / Average": k, **rep_dict[k]})
        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)
        st.caption(f"Snapshot policy: `{exp['snapshot_params']}` · Dataset version: `{exp['dataset_version']}` · "
                   f"Session timeout: `{exp['session_timeout_minutes']} min`")

    imp = exp.get("importance", {})
    i1, i2 = st.columns(2)
    with i1, ui.chart_card("Model-native feature importance / coefficients",
                           imp.get("native_kind") or "Not available"):
        if imp.get("native"):
            signed = "coefficient" in (imp.get("native_kind") or "").lower()
            st.plotly_chart(charts.importance_fig(imp["native"], signed=signed), use_container_width=True)
        else:
            st.caption("No native feature importance available for this estimator.")
    with i2, ui.chart_card("Permutation importance on held-out test set",
                           "Mean drop in PR-AUC when each feature is randomly permuted (model-agnostic)"):
        perm = (imp.get("permutation") or {}).get("mean")
        if perm:
            st.plotly_chart(charts.importance_fig(perm, signed=True), use_container_width=True)
        else:
            st.caption("Permutation importance unavailable.")

    with st.expander("Full saved configuration & metadata (JSON)"):
        st.json({k: v for k, v in exp.items() if k != "metrics"})
