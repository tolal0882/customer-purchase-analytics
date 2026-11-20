"""
Page 5 – Model Training & Evaluation:
  - Interactive Classification Training (Algorithm selection, test split ratio,
    chronological vs stratified split, random seed, CV folds, feature selection,
    class weighting, training-fold sampling, decision threshold, custom hyperparameters)
  - Full Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
    Confusion Matrix, ROC Curve, Precision-Recall Curve, Classification Report,
    Feature Importance, and Saved Experiment Comparison)
  - Unsupervised K-Means Clustering Performance (Elbow & Silhouette).
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import api_client
try:
    from components import react_bridge
except ImportError:
    from frontend.components import react_bridge

st.title("📈 Interactive Model Training & Evaluation")
st.caption(
    "Train and compare supervised classification pipelines with full hyperparameter control and zero target leakage, "
    "and inspect held-out test metrics, curves, classification reports, and clustering performance."
)

try:
    schema = api_client.get_schema()
    report = api_client.get_metrics()
    seg = api_client.get_segments(scope="customer")
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

all_feature_names = [f["name"] for f in schema["features"]]

# =============================================================================
# 1. INTERACTIVE CLASSIFICATION TRAINING STUDIO
# =============================================================================
with st.container(border=True):
    st.subheader("🧠 1. Interactive Model Training Studio")
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
    balanced = bool(c6.checkbox("Balanced Class Weighting", value=True, help="Weights classes inversely to frequency in the training fold."))
    sampling = c7.selectbox(
        "Training-Fold Sampling",
        ["none", "undersample", "oversample"],
        format_func=lambda s: {"none": "None", "undersample": "Random Under-Sampling", "oversample": "Random Over-Sampling"}[s],
        help="Applied strictly inside training folds via TrainingFoldResampler; test data is never resampled.",
    )
    threshold = float(c8.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.05))

    sel_features = st.multiselect(
        "Selected Input Features (IDs & post-purchase variables are blocked)",
        options=all_feature_names,
        default=schema.get("active_input_columns") or all_feature_names,
    )

    # Algorithm-specific hyperparameter controls
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

    if st.button("🚀 Train & Evaluate Model Pipeline", type="primary", use_container_width=True, disabled=not sel_features):
        prog = st.progress(0.15, text="Sending training configuration to FastAPI backend...")
        try:
            model_arg = "all" if algo.startswith("Compare") else algo
            prog.progress(0.45, text=f"Running {cv_folds}-fold CV & fitting {algo} on training window...")
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
            prog.progress(1.0, text="Training & held-out evaluation complete!")
            st.success(
                f"Trained **{report['best_model']}** (`{report.get('experiment_id', '')}`) · "
                f"Train: `{report['n_train']:,}` rows · Test: `{report['n_test']:,}` rows · "
                f"Test F1 = `{report['test_metrics']['f1']:.4f}` · ROC-AUC = `{report['test_metrics']['roc_auc']:.4f}`"
            )
            st.rerun()
        except api_client.APIError as e:
            prog.empty()
            st.error(str(e))

tab_eval, tab_exps, tab_unsup = st.tabs([
    "✅ 2. Active Model Evaluation (Test Set)",
    "🗂️ 3. Saved Experiments Comparison",
    "👥 4. Unsupervised Clustering Performance",
])

# =============================================================================
# 2. TAB 1 – ACTIVE MODEL EVALUATION (METRICS, CONFUSION MATRIX, ROC/PR, REPORT)
# =============================================================================
with tab_eval:
    tcfg = report.get("training_config", {})
    st.subheader(f"Active Model: **{report['best_model']}** (`{report.get('experiment_id', 'default')}`)")
    st.caption(
        f"Timestamp: `{report['trained_at']}` · Train samples: **{report['n_train']:,}** "
        f"(fitted after sampling: **{report.get('n_fitted_after_sampling', report['n_train']):,}**) · "
        f"Test samples: **{report['n_test']:,}** · Split: `{tcfg.get('split_mode', 'stratified')}` · "
        f"Threshold: `{tcfg.get('threshold', 0.5):.2f}` · Params: `{report['best_params'] or 'default'}`"
    )

    tm = report["test_metrics"]
    m_cols = st.columns(len(tm))
    for col, (name, value) in zip(m_cols, tm.items()):
        col.metric(name.upper().replace("_", "-"), f"{value:.4f}")

    st.info(
        "**Why Accuracy Alone Is Misleading (Precision–Recall Trade-off):** "
        f"Only **{report.get('positive_rate', 0.076) * 100:.2f}%** of eligible sessions convert into a purchase. "
        "A naive classifier predicting 'No Purchase' for every session would achieve ~92.4% accuracy while catching 0% of buyers! "
        "Class weighting and threshold tuning prioritize **Recall** (catching real buyers) and **F1 / PR-AUC**."
    )

    if len(report.get("cv_results", [])) > 0:
        with st.container(border=True):
            st.subheader("Cross-Validation Comparison on Training Set")
            if react_bridge.is_available() and len(report["cv_results"]) > 1:
                react_bridge.model_compare(
                    report["cv_results"],
                    best_model=report["best_model"],
                    primary_metric=report["primary_metric"],
                )
            st.dataframe(pd.DataFrame(report["cv_results"]), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    if report.get("confusion_matrix"):
        with c1, st.container(border=True):
            st.subheader("Confusion Matrix (Held-Out Test Set)")
            cm = report["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm["matrix"],
                index=[f"Actual {l}" for l in cm["labels"]],
                columns=[f"Predicted {l}" for l in cm["labels"]],
            )
            fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Greens", aspect="auto")
            fig_cm.update_layout(height=310, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

    if report.get("feature_importance"):
        with c2, st.container(border=True):
            st.subheader("Feature Importance (Final Pipeline)")
            imp_s = pd.Series(report["feature_importance"]).sort_values(ascending=True)
            fig_imp = px.bar(
                x=imp_s.values, y=imp_s.index, orientation="h",
                labels={"x": "Importance / |Coefficient|", "y": "Feature"},
                color_discrete_sequence=["#2F855A"],
            )
            fig_imp.update_layout(height=310, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

    curves = report.get("curves", {})
    if curves.get("roc") and curves.get("pr"):
        r1, r2 = st.columns(2)
        with r1, st.container(border=True):
            st.subheader(f"ROC Curve (ROC-AUC = {tm.get('roc_auc', 0):.4f})")
            fig_roc = go.Figure()
            fig_roc.add_scatter(x=curves["roc"]["fpr"], y=curves["roc"]["tpr"], mode="lines", name=report["best_model"], line=dict(color="#2F855A", width=2.5))
            fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (0.50)", line=dict(color="grey", dash="dash"))
            fig_roc.update_layout(height=300, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_roc, use_container_width=True)

        with r2, st.container(border=True):
            st.subheader(f"Precision–Recall Curve (PR-AUC = {tm.get('pr_auc', 0):.4f})")
            fig_pr = go.Figure()
            fig_pr.add_scatter(x=curves["pr"]["recall"], y=curves["pr"]["precision"], mode="lines", name=report["best_model"], line=dict(color="#153E2D", width=2.5))
            prev = report.get("positive_rate", 0.076)
            fig_pr.add_scatter(x=[0, 1], y=[prev, prev], mode="lines", name=f"Prevalence ({prev*100:.1f}%)", line=dict(color="grey", dash="dash"))
            fig_pr.update_layout(height=300, xaxis_title="Recall", yaxis_title="Precision", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pr, use_container_width=True)

    if report.get("classification_report"):
        with st.container(border=True):
            st.subheader("Full Classification Report (Held-Out Test Set)")
            cr = report["classification_report"]
            cr_rows = [
                {"Class / Average": k, **{m: round(float(v), 4) for m, v in cr[k].items()}}
                for k in ("No", "Yes", "macro avg", "weighted avg")
                if k in cr and isinstance(cr[k], dict)
            ]
            st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)

# =============================================================================
# 3. TAB 2 – SAVED EXPERIMENTS COMPARISON
# =============================================================================
with tab_exps:
    exps = report.get("experiments", [])
    if not exps:
        st.info("Train one or more models above to compare saved experiments.")
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
        exp_df["label"] = exp_df["model"] + " (" + exp_df["experiment_id"].astype(str).str[-6:] + ")"
        long_exp = exp_df.head(8).melt(
            id_vars=["label"],
            value_vars=["f1", "roc_auc", "pr_auc", "recall", "precision", "accuracy"],
            var_name="Metric",
            value_name="Score",
        )
        fig_cmp = px.bar(long_exp, x="Metric", y="Score", color="label", barmode="group", text_auto=".3f")
        fig_cmp.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.dataframe(exp_df.drop(columns=["label"]), use_container_width=True, hide_index=True)

# =============================================================================
# 4. TAB 3 – UNSUPERVISED K-MEANS CLUSTERING PERFORMANCE
# =============================================================================
with tab_unsup:
    st.subheader("K-Means Clustering Evaluation (K = 2 to 8)")
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Selected Clusters (K)", seg["n_clusters"])
    u2.metric("Silhouette Score", f"{seg['silhouette_score']:.4f}")
    u3.metric("Davies–Bouldin Index", f"{seg['davies_bouldin_index']:.4f}")
    u4.metric("Within-Cluster Inertia", f"{seg['inertia']:,.0f}")

    k_df = pd.DataFrame(seg.get("k_evaluation", []))
    if not k_df.empty:
        e1, e2 = st.columns(2)
        with e1, st.container(border=True):
            st.subheader("Elbow Method (Inertia vs K)")
            fig_el = px.line(k_df, x="k", y="inertia", markers=True, color_discrete_sequence=["#2F855A"])
            fig_el.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_el, use_container_width=True)
        with e2, st.container(border=True):
            st.subheader("Silhouette Score by K")
            fig_sil = px.bar(k_df, x="k", y="silhouette", text="silhouette", color_discrete_sequence=["#A7C957"])
            fig_sil.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_sil.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_sil, use_container_width=True)
        st.dataframe(k_df, use_container_width=True, hide_index=True)
