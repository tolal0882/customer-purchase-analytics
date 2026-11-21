"""
Model Training & Evaluation — Interactive classification training studio,
held-out test evaluation, React ModelCompare leaderboard, ROC & PR curves,
and saved experiment comparison.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    seg = api_client.get_segments(scope="customer")
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

tm = report["test_metrics"]
tcfg = report.get("training_config", {})
ui.page_header(
    "Model Training & Evaluation Studio",
    "Train and compare supervised early-session purchase classifiers with zero target leakage, and evaluate held-out test metrics and ROC/PR curves.",
    badges=[
        ("👤 Handled by: Vy Soksamnang", "ok"),
        (f"Active Winner: {report['best_model']}", "info"),
        (f"Test ROC-AUC: {tm.get('roc_auc', 0):.4f} · F1: {tm.get('f1', 0):.4f}", "neutral"),
    ],
)

all_feature_names = [f["name"] for f in schema["features"]]

tab_eval, tab_train, tab_exps, tab_unsup = st.tabs([
    "📊 Held-Out Test Evaluation",
    "🧠 Interactive Model Training Studio",
    "🗂️ Saved Experiments Registry",
    "👥 Unsupervised Clustering Evaluation",
])

# =============================================================================
# TAB 1: HELD-OUT TEST EVALUATION (ACTIVE MODEL)
# =============================================================================
with tab_eval:
    ui.kpi_row([
        ("Active Model", report["best_model"], f"Split: {tcfg.get('split_mode', 'stratified')} · Threshold: {tcfg.get('threshold', 0.5):.2f}"),
        ("Train / Test Rows", f"{report['n_train']:,} / {report['n_test']:,}", f"Fitted rows: {report.get('n_fitted_after_sampling', report['n_train']):,}"),
        ("ROC-AUC", f"{tm.get('roc_auc', 0):.4f}", "Area under ROC curve"),
        ("PR-AUC", f"{tm.get('pr_auc', 0):.4f}", f"Base prevalence: {report.get('positive_rate', 0.0756)*100:.2f}%"),
        ("F1-Score (Positive)", f"{tm.get('f1', 0):.4f}", "Harmonic mean of Precision & Recall"),
        ("Recall (Sensitivity)", f"{tm.get('recall', 0):.4f}", "Share of actual buyers detected"),
        ("Precision", f"{tm.get('precision', 0):.4f}", "Purity of predicted buyers"),
        ("Accuracy", f"{tm.get('accuracy', 0):.4f}", "Overall classification accuracy"),
    ])

    ui.note(
        f"<b>Class Imbalance & Precision–Recall Trade-off:</b> Only <b>{report.get('positive_rate', 0.0756)*100:.2f}%</b> "
        "of eligible sessions convert into a purchase. Accuracy alone is misleading because an all-negative classifier "
        "scores ~92.4% accuracy with 0% recall. We prioritize <b>F1, PR-AUC, and Recall</b> via balanced class weights."
    )

    if len(report.get("cv_results", [])) > 0:
        with ui.chart_card("Cross-Validation Leaderboard (Training Set)", "Interactive React comparison across candidate classifiers"):
            if react_bridge.is_available() and len(report["cv_results"]) > 1:
                react_bridge.model_compare(
                    report["cv_results"],
                    best_model=report["best_model"],
                    primary_metric=report["primary_metric"],
                )
            st.dataframe(pd.DataFrame(report["cv_results"]), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    if report.get("confusion_matrix"):
        with c1, ui.chart_card("Confusion Matrix (Held-Out Test Set)", f"Evaluated on {report['n_test']:,} unseen test sessions"):
            cm = report["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm["matrix"],
                index=[f"Actual {l}" for l in cm["labels"]],
                columns=[f"Predicted {l}" for l in cm["labels"]],
            )
            fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Greens", aspect="auto")
            fig_cm.update_layout(height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

    if report.get("feature_importance"):
        with c2, ui.chart_card("Feature Importance (Active Pipeline)", "Standardized coefficient magnitude or tree importance"):
            imp_s = pd.Series(report["feature_importance"]).sort_values(ascending=True)
            fig_imp = px.bar(
                x=imp_s.values, y=imp_s.index, orientation="h",
                labels={"x": "Importance / |Coefficient|", "y": "Feature"},
                color_discrete_sequence=["#2F855A"],
            )
            fig_imp.update_layout(height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

    curves = report.get("curves", {})
    if curves.get("roc") and curves.get("pr"):
        r1, r2 = st.columns(2)
        with r1, ui.chart_card(f"ROC Curve (ROC-AUC = {tm.get('roc_auc', 0):.4f})", "True positive rate vs false positive rate"):
            fig_roc = go.Figure()
            fig_roc.add_scatter(x=curves["roc"]["fpr"], y=curves["roc"]["tpr"], mode="lines", name=report["best_model"], line=dict(color="#2F855A", width=2.5))
            fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (0.50)", line=dict(color="grey", dash="dash"))
            fig_roc.update_layout(height=290, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_roc, use_container_width=True)

        with r2, ui.chart_card(f"Precision–Recall Curve (PR-AUC = {tm.get('pr_auc', 0):.4f})", "Precision vs Recall across decision thresholds"):
            fig_pr = go.Figure()
            fig_pr.add_scatter(x=curves["pr"]["recall"], y=curves["pr"]["precision"], mode="lines", name=report["best_model"], line=dict(color="#153E2D", width=2.5))
            prev = report.get("positive_rate", 0.0756)
            fig_pr.add_scatter(x=[0, 1], y=[prev, prev], mode="lines", name=f"Baseline ({prev*100:.1f}%)", line=dict(color="grey", dash="dash"))
            fig_pr.update_layout(height=290, xaxis_title="Recall", yaxis_title="Precision", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pr, use_container_width=True)

    if report.get("classification_report"):
        with ui.chart_card("Full Classification Report", "Per-class Precision, Recall, F1-score, and support on the held-out test set"):
            cr = report["classification_report"]
            cr_rows = [
                {"Class / Average": k, **{m: round(float(v), 4) for m, v in cr[k].items()}}
                for k in ("No", "Yes", "macro avg", "weighted avg")
                if k in cr and isinstance(cr[k], dict)
            ]
            st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: INTERACTIVE MODEL TRAINING STUDIO
# =============================================================================
with tab_train:
    with ui.chart_card(
        "Configure & Train Supervised Classification Pipeline",
        "All scaling, encoding, and optional resampling happen strictly inside training folds (zero test leakage)",
    ):
        c1, c2, c3, c4 = st.columns(4)
        algo = c1.selectbox(
            "Algorithm Selection",
            [
                "Compare All 4 & Auto-Tune Best",
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
            ],
        )
        split_label = c2.selectbox("Train/Test Split Mode", ["Chronological (Time-Ordered)", "Stratified Random"])
        split_mode = "chronological" if split_label.startswith("Chronological") else "stratified"
        test_ratio = float(c3.slider("Test Split Ratio", 0.10, 0.40, 0.20, 0.05))
        cv_folds = int(c4.slider("Cross-Validation Folds", 2, 10, 5))

        c5, c6, c7, c8 = st.columns(4)
        seed = int(c5.number_input("Random Seed", 0, 9999, 42))
        balanced = bool(c6.checkbox("Balanced Class Weighting", value=True))
        sampling = c7.selectbox(
            "Training-Fold Resampling",
            ["none", "undersample", "oversample"],
            format_func=lambda s: {"none": "None", "undersample": "Random Under-Sampling", "oversample": "Random Over-Sampling"}[s],
        )
        threshold = float(c8.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.05))

        sel_features = st.multiselect(
            "Selected Input Features (IDs & post-purchase fields are blocked)",
            options=all_feature_names,
            default=schema.get("active_input_columns") or all_feature_names,
        )

        hp: dict = {}
        if algo == "Logistic Regression":
            hp["C"] = float(st.select_slider("Inverse Regularization Strength (C)", options=[0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0], value=1.0))
        elif algo == "Decision Tree":
            h1, h2, h3 = st.columns(3)
            hp["max_depth"] = int(h1.slider("Max Tree Depth", 2, 20, 6))
            hp["min_samples_leaf"] = int(h2.slider("Min Samples per Leaf", 1, 200, 20))
            hp["criterion"] = h3.selectbox("Split Criterion", ["gini", "entropy"])
        elif algo == "Random Forest":
            h1, h2, h3 = st.columns(3)
            hp["n_estimators"] = int(h1.slider("Number of Trees (n_estimators)", 50, 300, 150, 25))
            hp["max_depth"] = int(h2.slider("Max Tree Depth", 4, 25, 12))
            hp["min_samples_leaf"] = int(h3.slider("Min Samples per Leaf", 1, 100, 10))
        elif algo == "Gradient Boosting":
            h1, h2, h3 = st.columns(3)
            hp["n_estimators"] = int(h1.slider("Boosting Stages (n_estimators)", 50, 300, 120, 20))
            hp["learning_rate"] = float(h2.select_slider("Learning Rate", options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.1))
            hp["max_depth"] = int(h3.slider("Individual Tree Depth", 2, 8, 3))

        if st.button("🚀 Train & Save Classification Pipeline", type="primary", use_container_width=True, disabled=not sel_features):
            prog = st.progress(0.2, text="Running cross-validation and fitting pipeline...")
            try:
                model_arg = "all" if algo.startswith("Compare") else algo
                report = api_client.train({
                    "model_name": model_arg,
                    "test_size": test_ratio,
                    "random_state": seed,
                    "cv_folds": cv_folds,
                    "selected_features": sel_features,
                    "balanced": balanced,
                    "sampling": sampling,
                    "threshold": threshold,
                    "split_mode": split_mode,
                    "hyperparams": hp,
                })
                prog.progress(1.0, text="Complete!")
                st.success(
                    f"Trained **{report['best_model']}** (`{report.get('experiment_id', '')}`) · "
                    f"Test F1 = `{report['test_metrics']['f1']:.4f}` · ROC-AUC = `{report['test_metrics']['roc_auc']:.4f}`"
                )
                st.rerun()
            except api_client.APIError as e:
                prog.empty()
                st.error(str(e))

# =============================================================================
# TAB 3: SAVED EXPERIMENTS REGISTRY
# =============================================================================
with tab_exps:
    exps = report.get("experiments", [])
    if not exps:
        ui.empty_state("No saved experiments yet", "Train a model in the Training Studio tab.", "🧪")
    else:
        exp_rows = []
        for e in exps:
            m = e.get("test_metrics", {})
            tc = e.get("training_config", {})
            exp_rows.append({
                "experiment_id": e.get("experiment_id", "default"),
                "trained_at": e.get("trained_at"),
                "model": e.get("best_model"),
                "split_mode": tc.get("split_mode", "stratified"),
                "sampling": tc.get("sampling", "none"),
                "threshold": tc.get("threshold", 0.5),
                "f1": m.get("f1"),
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "recall": m.get("recall"),
                "precision": m.get("precision"),
                "accuracy": m.get("accuracy"),
            })
        exp_df = pd.DataFrame(exp_rows)
        with ui.chart_card("Saved Experiments Metric Comparison", "Side-by-side test metrics across logged training runs"):
            exp_df["label"] = exp_df["model"] + " (" + exp_df["experiment_id"].astype(str).str[-6:] + ")"
            long_exp = exp_df.head(8).melt(
                id_vars=["label"],
                value_vars=["f1", "roc_auc", "pr_auc", "recall", "precision", "accuracy"],
                var_name="Metric",
                value_name="Score",
            )
            fig_cmp = px.bar(long_exp, x="Metric", y="Score", color="label", barmode="group", text_auto=".3f")
            fig_cmp.update_layout(height=320, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.dataframe(exp_df.drop(columns=["label"]), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4: UNSUPERVISED CLUSTERING PERFORMANCE
# =============================================================================
with tab_unsup:
    ui.kpi_row([
        ("Selected Clusters (K)", str(seg["n_clusters"]), "Customer K-Means"),
        ("Silhouette Score", f"{seg['silhouette_score']:.4f}", "Cluster cohesion & separation"),
        ("Davies–Bouldin Index", f"{seg['davies_bouldin_index']:.4f}", "Lower values indicate tighter clusters"),
        ("Within-Cluster Inertia", f"{seg['inertia']:,.0f}", "Sum of squared distances"),
    ])
    k_df = pd.DataFrame(seg.get("k_evaluation", []))
    if not k_df.empty:
        e1, e2 = st.columns(2)
        with e1, ui.chart_card("Elbow Curve (Inertia vs K)"):
            fig_el = px.line(k_df, x="k", y="inertia", markers=True, color_discrete_sequence=["#2F855A"])
            fig_el.update_layout(height=280, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_el, use_container_width=True)
        with e2, ui.chart_card("Silhouette Score by K"):
            fig_sil = px.bar(k_df, x="k", y="silhouette", text="silhouette", color_discrete_sequence=["#A7C957"])
            fig_sil.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_sil.update_layout(height=280, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_sil, use_container_width=True)
