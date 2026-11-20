"""Page 6: Model Training (leakage-safe early-session purchase classification)."""
import pandas as pd
import streamlit as st

from app_pages import common
from components.streamlit_components import ui
from config import settings
from src import charts, classification as clf, feature_engineering as fe, model_manager


def _hyperparam_controls(model_name: str) -> dict:
    specs = clf.HYPERPARAM_SPECS[model_name]
    cols = st.columns(max(1, len(specs)))
    out = {}
    for col, sp in zip(cols, specs):
        k = f"hp_{model_name}_{sp['name']}"
        if sp["type"] == "int":
            out[sp["name"]] = int(col.number_input(sp["label"], int(sp["min"]), int(sp["max"]), int(sp["default"]), key=k))
        elif sp["type"] == "float":
            out[sp["name"]] = float(col.number_input(sp["label"], float(sp["min"]), float(sp["max"]), float(sp["default"]),
                                                     format="%.4f", key=k))
        elif sp["type"] == "choice":
            out[sp["name"]] = col.selectbox(sp["label"], sp["options"], index=sp["options"].index(sp["default"]), key=k)
    return out


def render():
    ui.page_header(
        "Model Training",
        "Train and save early-session purchase classifiers with chronological splitting, expanding-window cross-validation, and zero target leakage.",
    )
    core = common.require_core()
    meta_ds = core.meta

    with ui.chart_card("1 · Prediction snapshot policy (leakage prevention)",
                       "Features are computed ONLY from events observed at or before the snapshot. Target = 1 iff a transaction occurs strictly after the snapshot."):
        p1, p2, p3 = st.columns([1.2, 1, 1])
        policy_label = p1.radio(
            "Snapshot cutoff mode",
            ["After N non-transaction events", "Time elapsed since session start"],
            horizontal=True,
            key="mt_policy",
        )
        policy = "n_events" if policy_label.startswith("After") else "time"
        n_ev = int(p2.number_input("N eligible events", 1, 20, settings.DEFAULT_SNAPSHOT_EVENTS,
                                   disabled=(policy != "n_events"), key="mt_nev"))
        cut_min = float(p3.number_input("Time cutoff (minutes)", 0.5, 60.0, settings.DEFAULT_SNAPSHOT_CUTOFF_MIN, 0.5,
                                        disabled=(policy != "time"), key="mt_cut"))

        with ui.loading_state("Building prediction snapshots…"):
            snap_res = common.get_snapshots(policy, n_ev, cut_min)
        rep = snap_res.report
        snap_df = snap_res.data

        ui.kpi_row([
            ("Total sessions", common.fmt_int(rep["total_sessions"]), f"{meta_ds['session_timeout_minutes']:g}-min timeout"),
            ("Usable prediction samples", common.fmt_int(rep["eligible_samples"]), f"{rep['eligible_samples'] / max(rep['total_sessions'], 1) * 100:.2f}% of sessions"),
            ("Positive samples (Purchase = 1)", common.fmt_int(rep["positives"]), f"positive rate {rep['positive_rate'] * 100:.2f}%"),
            ("Negative samples (No Purchase = 0)", common.fmt_int(rep["negatives"]), f"{100 - rep['positive_rate'] * 100:.2f}%"),
        ])

        with st.expander("Why are some sessions excluded? (Sample eligibility breakdown)"):
            ex_df = pd.DataFrame([
                {"Exclusion reason": "Insufficient activity before snapshot",
                 "All sessions excluded": rep["excluded_insufficient_prefix"],
                 "Purchase sessions excluded": rep["purchase_sessions_excluded_insufficient_prefix"],
                 "Methodological rationale": "Session ended before reaching the snapshot threshold."},
                {"Exclusion reason": "Transaction occurred at or before snapshot",
                 "All sessions excluded": rep["excluded_purchase_before_or_at_snapshot"],
                 "Purchase sessions excluded": rep["purchase_sessions_excluded_purchase_before_snapshot"],
                 "Methodological rationale": "Outcome already happened; predicting it would be post-outcome leakage."},
                {"Exclusion reason": "No events after snapshot (no prediction window)",
                 "All sessions excluded": rep["excluded_no_future_window"],
                 "Purchase sessions excluded": rep["purchase_sessions_excluded_no_future_window"],
                 "Methodological rationale": "No future window remains in the session after the snapshot."},
            ])
            st.dataframe(ex_df, use_container_width=True, hide_index=True)
            ui.note("Because only ~7.5% of eligible sessions convert after the snapshot, <b>accuracy alone is misleading</b> "
                    "(a trivial all-negative classifier achieves ~92.4% accuracy). Always compare PR-AUC, F1, Recall, and Precision.")

    avail_feats = fe.available_snapshot_features(snap_df)
    default_feats = [f for f in avail_feats if not f.startswith("hist_")]
    if policy == "n_events" and "total_events" in default_feats:
        default_feats = [f for f in default_feats if f != "total_events"]

    with ui.chart_card("2 · Model & training configuration",
                       "All preprocessing (median imputation, StandardScaler, optional resampling) is wrapped inside a Pipeline and fitted on training folds only."):
        c1, c2, c3, c4 = st.columns(4)
        model_name = c1.selectbox("Algorithm", clf.MODEL_NAMES, key="mt_algo")
        test_ratio = float(c2.slider("Chronological test split ratio", 0.10, 0.40, settings.DEFAULT_TEST_RATIO, 0.05, key="mt_test"))
        cv_folds = int(c3.slider("TimeSeriesSplit CV folds", 2, 5, 3, key="mt_cv"))
        seed = int(c4.number_input("Random seed", 0, 10_000, settings.DEFAULT_RANDOM_SEED, key="mt_seed"))

        c5, c6, c7, c8 = st.columns(4)
        balanced = c5.checkbox("Balanced class weighting", value=True, key="mt_bal",
                               help="Weights classes inversely proportional to class frequency in the training fold.")
        sampling = c6.selectbox("Training-fold sampling", list(clf.SAMPLING_METHODS.keys()),
                                format_func=lambda k: clf.SAMPLING_METHODS[k], key="mt_samp",
                                help="Applied strictly inside training folds via ResamplingClassifier; validation and test sets are never resampled.")
        samp_ratio = float(c7.slider("Target minority/majority ratio", 0.1, 1.0, 0.5, 0.1,
                                     disabled=(sampling == "none"), key="mt_sratio"))
        threshold = float(c8.slider("Decision probability threshold", 0.10, 0.90, 0.50, 0.05, key="mt_thr"))

        feats = st.multiselect("Selected pre-snapshot features (forbidden IDs & post-purchase variables are blocked)",
                               avail_feats, default=default_feats, key="mt_feats")
        with st.expander("Feature definitions & leakage status"):
            f_rows = [{"Feature": f, "Selected": "✅" if f in feats else "—",
                       "Description": fe.FEATURE_DESCRIPTIONS.get(f, "")} for f in avail_feats]
            st.dataframe(pd.DataFrame(f_rows), use_container_width=True, hide_index=True)

        st.markdown("**Hyperparameters**")
        hp = _hyperparam_controls(model_name)
        note_txt = st.text_input("Experiment note (optional)", placeholder="e.g. Baseline balanced model after 3 events", key="mt_note")

        b1, b2 = st.columns([1, 1])
        run_one = b1.button(f"🚀 Train & Save {model_name}", type="primary", use_container_width=True, disabled=not feats)
        run_all = b2.button("⚡ Train & Compare All 4 Algorithms (default hyperparameters)", use_container_width=True, disabled=not feats)

    if not feats:
        ui.empty_state("No features selected", "Select at least one pre-snapshot feature above to enable training.", "🧱")
        return

    if run_one:
        bar = st.progress(0.0, text=f"Training {model_name}…")
        cfg = clf.TrainConfig(
            model_name=model_name, features=feats, test_ratio=test_ratio, seed=seed,
            cv_folds=cv_folds, balanced=balanced, hyperparams=hp,
            sampling=sampling, sampling_ratio=samp_ratio, threshold=threshold,
        )
        res = clf.train_model(snap_df, cfg, progress=lambda frac, msg: bar.progress(frac, text=msg))
        saved = model_manager.save_experiment(
            res, snap_res.params, rep, meta_ds["dataset_version"], meta_ds["session_timeout_minutes"], note=note_txt
        )
        model_manager.set_active_experiment(saved["experiment_id"])
        bar.empty()
        st.session_state["last_trained_id"] = saved["experiment_id"]
        st.success(f"Trained and saved `{saved['experiment_id']}` in {saved['train_seconds']}s (set as active model).")

    if run_all:
        bar = st.progress(0.0, text="Training all 4 algorithms…")
        total = len(clf.MODEL_NAMES)
        last_id = None
        for idx, mname in enumerate(clf.MODEL_NAMES):
            cfg = clf.TrainConfig(
                model_name=mname, features=feats, test_ratio=test_ratio, seed=seed,
                cv_folds=cv_folds, balanced=balanced, hyperparams=clf.default_hyperparams(mname),
                sampling=sampling, sampling_ratio=samp_ratio, threshold=threshold,
            )
            res = clf.train_model(
                snap_df, cfg,
                progress=lambda frac, msg, i=idx, n=mname: bar.progress((i + frac) / total, text=f"[{n}] {msg}"),
            )
            saved = model_manager.save_experiment(
                res, snap_res.params, rep, meta_ds["dataset_version"], meta_ds["session_timeout_minutes"],
                note=note_txt or "Batch benchmark",
            )
            last_id = saved["experiment_id"]
        if last_id:
            model_manager.set_active_experiment(last_id)
            st.session_state["last_trained_id"] = last_id
        bar.empty()
        st.success("Trained and saved all 4 models. Compare them below or on the Model Evaluation page.")

    exps = model_manager.list_experiments()
    if not exps:
        ui.empty_state("No trained experiments yet", "Click **Train & Save** above to run your first experiment.", "🧠")
        return

    focus_id = st.session_state.get("last_trained_id") or exps[0]["experiment_id"]
    focus = next((e for e in exps if e["experiment_id"] == focus_id), exps[0])
    mt = focus["metrics"]
    sp = focus["split"]

    with ui.chart_card(f"3 · Latest experiment summary: {focus['model_name']} ({focus['experiment_id']})",
                       f"Trained at {focus['created_at']} in {focus['train_seconds']}s · dataset version {focus['dataset_version']}"):
        ui.kpi_row([
            ("Training samples", common.fmt_int(sp["train_samples"]), f"{common.fmt_int(sp['train_positives'])} positives (fitted: {common.fmt_int(sp['samples_fitted_after_sampling'])})"),
            ("Held-out test samples", common.fmt_int(sp["test_samples"]), f"{common.fmt_int(sp['test_positives'])} positives ({mt['prevalence'] * 100:.2f}%)"),
            ("PR-AUC", f"{mt['pr_auc']:.3f}", f"no-skill baseline {mt['prevalence']:.3f}"),
            ("ROC-AUC", f"{mt['roc_auc']:.3f}", "0.500 = random"),
            ("F1-score", f"{mt['f1']:.3f}", f"threshold {focus['threshold']:.2f}"),
            ("Precision", f"{mt['precision']:.3f}", f"{common.fmt_int(mt['predicted_positive'])} predicted positive"),
            ("Recall", f"{mt['recall']:.3f}", "share of true purchases caught"),
            ("Accuracy", f"{mt['accuracy']:.3f}", f"majority baseline {mt['majority_baseline_accuracy']:.3f}"),
        ])
        t_start = pd.to_datetime(sp["train_start_ts"], unit="ms")
        t_end = pd.to_datetime(sp["train_end_ts"], unit="ms")
        te_start = pd.to_datetime(sp["test_start_ts"], unit="ms")
        te_end = pd.to_datetime(sp["test_end_ts"], unit="ms")
        st.caption(f"Chronological windows — Train: `{t_start}` → `{t_end}` · Test: `{te_start}` → `{te_end}` (no future-to-past leakage).")
        if focus.get("cv"):
            st.markdown("**Expanding-window cross-validation folds (on training window)**")
            cv_df = pd.DataFrame(focus["cv"])
            st.dataframe(cv_df.round(4), use_container_width=True, hide_index=True)

    with ui.chart_card("4 · Saved experiment comparison", "All persisted runs in experiments/"):
        tbl = model_manager.experiments_table(exps)
        tbl["label"] = tbl["model"] + " (" + tbl["experiment_id"].str[-15:] + ")"
        st.plotly_chart(
            charts.model_compare_fig(tbl.head(8), ["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"]),
            use_container_width=True,
        )
        st.dataframe(tbl.drop(columns=["label"]).round(4), use_container_width=True, hide_index=True)
