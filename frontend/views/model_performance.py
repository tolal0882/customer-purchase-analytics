"""
Page 5 – Model Performance: compare all 4 supervised classifiers (5-fold CV),
inspect held-out test metrics, Confusion Matrix, ROC/PR curves, Feature Importance,
and evaluate K-Means clustering performance (Elbow Method & Silhouette Score).
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

st.title("📈 Model Performance (Supervised & Unsupervised)")
st.caption(
    "Compare the 4 classification algorithms on 5-fold Stratified Cross-Validation, evaluate the tuned model on the "
    "held-out test set, and inspect K-Means Elbow & Silhouette performance."
)

# =============================================================================
# 1. RETRAIN PIPELINE TRIGGER (POST /api/train)
# =============================================================================
with st.expander("🔄 Retrain Supervised & Unsupervised Models"):
    st.write(
        "Runs the complete `ml.train` pipeline on the backend: compares Logistic Regression, Decision Tree, "
        "Random Forest, and Gradient Boosting via 5-fold CV, tunes the winner with RandomizedSearchCV, "
        "evaluates on the held-out test set, and fits K-Means Customer Segmentation."
    )
    if st.button("🚀 Train Models Now", type="primary"):
        with st.spinner("Training classification & clustering pipelines on the backend..."):
            try:
                api_client.train()
                st.success("All models trained, evaluated, and saved!")
                st.rerun()
            except api_client.APIError as e:
                st.error(str(e))

try:
    report = api_client.get_metrics()
    seg = api_client.get_segments()
except api_client.APIError as e:
    if str(e).startswith("404"):
        st.info("No trained model yet. Open **Retrain Supervised & Unsupervised Models** above and click **Train Models Now**.")
        st.stop()
    api_client.show_error_and_stop(e)

tab_sup, tab_unsup = st.tabs(["🎯 Supervised: Purchase Prediction", "👥 Unsupervised: K-Means Clustering"])

# =============================================================================
# 2. TAB 1 – SUPERVISED CLASSIFICATION EVALUATION
# =============================================================================
with tab_sup:
    st.subheader(f"Selected Best Model: **{report['best_model']}**")
    st.caption(
        f"Target: `{report['target']}` · Trained: `{report['trained_at']}` · "
        f"Train rows: **{report['n_train']:,}** · Held-out test rows: **{report['n_test']:,}** · "
        f"Best tuned parameters: `{report['best_params'] or 'default'}`"
    )

    # Held-out test metrics cards
    tm = report["test_metrics"]
    cols = st.columns(len(tm))
    for col, (name, value) in zip(cols, tm.items()):
        col.metric(name.upper().replace("_", "-"), f"{value:.4f}")

    # Interactive React Leaderboard + 5-fold CV comparison table
    with st.container(border=True):
        st.subheader("Model Comparison (5-Fold Stratified Cross-Validation on Training Set)")
        if react_bridge.is_available():
            react_bridge.model_compare(
                report["cv_results"],
                best_model=report["best_model"],
                primary_metric=report["primary_metric"],
            )
        cv_df = pd.DataFrame(report["cv_results"])
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    # Confusion Matrix
    if report.get("confusion_matrix"):
        with c1, st.container(border=True):
            st.subheader("Confusion Matrix (Held-Out Test Set)")
            cm = report["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm["matrix"],
                index=[f"Actual {l}" for l in cm["labels"]],
                columns=[f"Predicted {l}" for l in cm["labels"]],
            )
            fig_cm = px.imshow(
                cm_df, text_auto=True, color_continuous_scale="Greens", aspect="auto"
            )
            fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

    # Feature Importance
    if report.get("feature_importance"):
        with c2, st.container(border=True):
            st.subheader("Feature Importance (Final Model)")
            imp_s = pd.Series(report["feature_importance"]).sort_values(ascending=True)
            fig_imp = px.bar(
                x=imp_s.values, y=imp_s.index, orientation="h",
                labels={"x": "Importance / |Coefficient|", "y": "Feature"},
                color_discrete_sequence=["#2F855A"],
            )
            fig_imp.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

    # ROC and Precision-Recall Curves
    curves = report.get("curves", {})
    if curves.get("roc") and curves.get("pr"):
        r1, r2 = st.columns(2)
        with r1, st.container(border=True):
            st.subheader(f"ROC Curve (Test ROC-AUC = {tm.get('roc_auc', 0):.4f})")
            fig_roc = go.Figure()
            fig_roc.add_scatter(x=curves["roc"]["fpr"], y=curves["roc"]["tpr"], mode="lines", name=report["best_model"], line=dict(color="#2F855A", width=2.5))
            fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline", line=dict(color="grey", dash="dash"))
            fig_roc.update_layout(height=320, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_roc, use_container_width=True)

        with r2, st.container(border=True):
            st.subheader(f"Precision–Recall Curve (Test PR-AUC = {tm.get('pr_auc', 0):.4f})")
            fig_pr = go.Figure()
            fig_pr.add_scatter(x=curves["pr"]["recall"], y=curves["pr"]["precision"], mode="lines", name=report["best_model"], line=dict(color="#153E2D", width=2.5))
            prev = report.get("positive_rate", 0.076)
            fig_pr.add_scatter(x=[0, 1], y=[prev, prev], mode="lines", name=f"Prevalence ({prev*100:.1f}%)", line=dict(color="grey", dash="dash"))
            fig_pr.update_layout(height=320, xaxis_title="Recall", yaxis_title="Precision", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pr, use_container_width=True)

# =============================================================================
# 3. TAB 2 – UNSUPERVISED K-MEANS CLUSTERING EVALUATION (ELBOW & SILHOUETTE)
# =============================================================================
with tab_unsup:
    st.subheader("K-Means Clustering Evaluation (K = 2 to 8)")
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Selected Clusters (K)", seg["n_clusters"])
    u2.metric("Silhouette Score", f"{seg['silhouette_score']:.4f}", "Higher = better separation")
    u3.metric("Davies–Bouldin Index", f"{seg['davies_bouldin_index']:.4f}", "Lower = more compact clusters")
    u4.metric("Within-Cluster Inertia", f"{seg['inertia']:,.0f}", "Sum of squared distances")

    k_df = pd.DataFrame(seg.get("k_evaluation", []))
    if not k_df.empty:
        e1, e2 = st.columns(2)
        with e1, st.container(border=True):
            st.subheader("Elbow Method (Inertia vs K)")
            fig_el = px.line(k_df, x="k", y="inertia", markers=True, color_discrete_sequence=["#2F855A"])
            fig_el.update_layout(height=320, xaxis_title="Number of Clusters (K)", yaxis_title="Inertia (SSE)", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_el, use_container_width=True)
        with e2, st.container(border=True):
            st.subheader("Silhouette Score by K")
            fig_sil = px.bar(k_df, x="k", y="silhouette", text="silhouette", color_discrete_sequence=["#A7C957"])
            fig_sil.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_sil.update_layout(height=320, xaxis_title="Number of Clusters (K)", yaxis_title="Silhouette Score", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_sil, use_container_width=True)

        with st.container(border=True):
            st.subheader("Clustering Evaluation Table Across K")
            st.dataframe(k_df, use_container_width=True, hide_index=True)
