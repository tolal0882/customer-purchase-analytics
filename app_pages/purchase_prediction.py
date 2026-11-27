"""
PAGE 4 — Purchase Prediction (Section 6 & Section 8):
  Combines:
    1. Model Training & Comparison (Logistic Regression, Decision Tree, Random Forest,
       Stratified 80/20 split, Stratified 5-fold CV, hyperparameters, class weighting)
    2. Model Evaluation (Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix,
       Classification Report, ROC Curve, Precision-Recall Curve, Feature Importance)
    3. PageValues Investigation (17 features with PageValues vs 16 features excluding PageValues)
    4. Interactive Session Purchase Prediction (using saved compatible Joblib pipeline)
    5. Local Prediction History & CSV Export (clearly distinguished from actual purchases)
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import classification, data_loader, data_validation, model_manager, prediction, ui

ui.inject_css()
df_raw = data_loader.require_dataset()

# Ensure a baseline trained model exists on first visit so evaluation and prediction work out-of-the-box
if model_manager.load_classifier_bundle() is None and "latest_training_results" not in st.session_state:
    with st.spinner("Training initial baseline classification pipelines (Logistic Regression, Decision Tree, Random Forest)..."):
        st.session_state["latest_training_results"] = classification.train_and_evaluate_models(
            df_raw,
            selected_algorithms=classification.ALGORITHMS,
            selected_features=data_validation.INPUT_FEATURES,
            test_size=0.20,
            random_state=42,
            cv_folds=5,
            use_class_weight=True,
        )

active_bundle = model_manager.load_classifier_bundle()
saved_models = model_manager.list_saved_classifiers()

ui.page_header(
    title="Purchase Intention Prediction (Supervised Learning)",
    subtitle="Train, compare, and evaluate Logistic Regression, Decision Tree, and Random Forest pipelines on Revenue, investigate PageValues, and run session predictions.",
    category="Page 4 · Supervised Classification (Task 1)",
    badges=[
        (f"Active Model: {active_bundle['model_name']}" if active_bundle else "No Model Trained", "ok" if active_bundle else "warn"),
        ("Stratified 80/20 Split + 5-Fold CV", "info"),
        ("Target: Revenue (0 = No, 1 = Yes)", "neutral"),
    ],
)

tab_train, tab_pv, tab_pred, tab_hist = st.tabs([
    ":material/model_training: 1. Model Training, Comparison & Evaluation",
    ":material/science: 2. PageValues Investigation (17 vs. 16 Features)",
    ":material/online_prediction: 3. Interactive Purchase Prediction",
    ":material/history: 4. Local Prediction History",
])

# =============================================================================
# TAB 1: MODEL TRAINING, COMPARISON & EVALUATION
# =============================================================================
with tab_train:
    with ui.chart_card(
        "Configure & Train Supervised Classification Pipelines",
        "All preprocessing (imputation, scaling, one-hot encoding) is fitted strictly on the training split inside a Scikit-learn Pipeline.",
    ):
        c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
        sel_algos = c1.multiselect(
            "Select Classification Algorithms (1 to 3)",
            options=classification.ALGORITHMS,
            default=classification.ALGORITHMS,
        )
        test_pct = int(c2.slider("Test Split (%)", min_value=10, max_value=40, value=20, step=5))
        seed_val = int(c3.number_input("Random Seed", min_value=0, max_value=9999, value=42))
        cv_folds = int(c4.slider("Stratified CV Folds", min_value=3, max_value=10, value=5))

        c5, c6, c7, c8 = st.columns(4)
        use_cw = c5.checkbox("Balanced Class Weighting", value=True, help="Sets class_weight='balanced' to address the 15.47% positive class imbalance.")
        drop_dups = c6.checkbox("Remove 125 Exact Duplicates", value=bool(st.session_state.get("prep_drop_duplicates", False)))
        scaler_name = c7.selectbox("Numerical Scaler", ["standard", "robust", "minmax"], index=0)
        primary_metric = c8.selectbox("Primary Ranking Metric", ["f1", "roc_auc", "pr_auc", "recall", "precision"], index=0)

        sel_features = st.multiselect(
            "Selected Input Features (Revenue is strictly excluded from inputs)",
            options=data_validation.INPUT_FEATURES,
            default=data_validation.INPUT_FEATURES,
        )

        with st.expander("Configure Algorithm Hyperparameters", icon=":material/settings:", expanded=False):
            hp_col1, hp_col2, hp_col3 = st.columns(3)
            with hp_col1:
                st.markdown("**Logistic Regression**")
                lr_c = float(st.select_slider("Regularization C", options=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0))
                lr_iter = int(st.number_input("Max Iterations", min_value=200, max_value=5000, value=1000, step=200))
            with hp_col2:
                st.markdown("**Decision Tree Classifier**")
                dt_depth = int(st.slider("DT Max Depth", min_value=2, max_value=20, value=6))
                dt_leaf = int(st.slider("DT Min Samples Leaf", min_value=1, max_value=100, value=10))
                dt_crit = st.selectbox("DT Criterion", ["gini", "entropy"], index=0)
            with hp_col3:
                st.markdown("**Random Forest Classifier**")
                rf_est = int(st.slider("RF n_estimators", min_value=50, max_value=300, value=150, step=25))
                rf_depth = int(st.slider("RF Max Depth", min_value=3, max_value=25, value=10))
                rf_leaf = int(st.slider("RF Min Samples Leaf", min_value=1, max_value=50, value=5))

        hyperparams_map = {
            "Logistic Regression": {"C": lr_c, "max_iter": lr_iter},
            "Decision Tree Classifier": {"max_depth": dt_depth, "min_samples_leaf": dt_leaf, "criterion": dt_crit},
            "Random Forest Classifier": {"n_estimators": rf_est, "max_depth": rf_depth, "min_samples_leaf": rf_leaf},
        }

        train_btn = st.button(
            "Train, Cross-Validate & Save Selected Models",
            icon=":material/play_arrow:",
            type="primary",
            use_container_width=True,
            disabled=(len(sel_algos) == 0 or len(sel_features) == 0),
        )

    if train_btn:
        with st.spinner(f"Training {len(sel_algos)} algorithm(s) with {cv_folds}-fold Stratified CV..."):
            res = classification.train_and_evaluate_models(
                df_raw,
                selected_algorithms=sel_algos,
                selected_features=sel_features,
                test_size=test_pct / 100.0,
                random_state=seed_val,
                cv_folds=cv_folds,
                use_class_weight=use_cw,
                scaler_name=scaler_name,
                drop_duplicates=drop_dups,
                hyperparams_by_algo=hyperparams_map,
                primary_metric=primary_metric,
                persist_best=True,
            )
            st.session_state["latest_training_results"] = res
            active_bundle = model_manager.load_classifier_bundle()
            saved_models = model_manager.list_saved_classifiers()
            st.success(
                f"Training complete! Best model by Test `{primary_metric.upper()}`: **{res['best_algorithm']}** "
                f"(saved to `models/classification/`)."
            )

    train_res = st.session_state.get("latest_training_results")
    if train_res:
        ds_info = train_res["dataset_info"]
        ui.kpi_row(
            [
                ("Best Algorithm", train_res["best_algorithm"], f"Ranked by Test {train_res['primary_metric'].upper()}"),
                ("Training Split (80%)", f"{ds_info['n_train']:,} sessions", "Fitted with Stratified K-Fold CV"),
                ("Held-Out Test Split (20%)", f"{ds_info['n_test']:,} sessions", f"Positive Revenue rate: {ds_info['positive_rate_pct']}%"),
            ],
            per_row=3,
        )

        with ui.chart_card("Model Comparison Table (Same Stratified Train/Test Split)", "Cross-validation and held-out test metrics across trained algorithms"):
            comp_df = pd.DataFrame(train_res["comparison_table"])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            long_m = comp_df.melt(
                id_vars=["Algorithm"],
                value_vars=["Test Precision", "Test Recall", "Test F1", "Test ROC-AUC", "Test PR-AUC", "Test Accuracy"],
                var_name="Metric",
                value_name="Score",
            )
            fig_cmp = px.bar(
                long_m,
                x="Metric",
                y="Score",
                color="Algorithm",
                barmode="group",
                text="Score",
                color_discrete_sequence=["#2F855A", "#153E2D", "#A7C957"],
            )
            fig_cmp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            ui.apply_plotly_theme(fig_cmp, height=320)
            st.plotly_chart(fig_cmp, use_container_width=True)

        ui.note(
            "<b>Why Precision, Recall, F1, and PR-AUC Matter for Purchase Prediction (Class Imbalance):</b> "
            "Because approximately <b>84.5%</b> of sessions do not result in a purchase, a trivial classifier that predicts "
            "<i>'No Purchase'</i> for every session achieves <b>84.5% accuracy</b> while detecting <b>0% of actual buyers (Recall = 0)</b>. "
            "<b>Recall</b> measures what fraction of true purchase-intent sessions the model successfully identifies, "
            "while <b>Precision</b> measures how many predicted buyers actually convert. <b>F1-score</b> and <b>PR-AUC</b> balance both."
        )

        # Detailed Evaluation Inspector for any trained model in the current run
        avail_models = list(train_res["models"].keys())
        inspected_algo = st.selectbox(
            "Select Trained Model to Inspect Detailed Evaluation & Curves",
            options=avail_models,
            index=avail_models.index(train_res["best_algorithm"]) if train_res["best_algorithm"] in avail_models else 0,
        )
        m_detail = train_res["models"][inspected_algo]
        tm = m_detail["test_metrics"]

        ui.kpi_row(
            [
                ("Test Precision", f"{tm['precision']:.4f}", "Positive predictive value"),
                ("Test Recall", f"{tm['recall']:.4f}", "Sensitivity on Purchase=True"),
                ("Test F1-Score", f"{tm['f1']:.4f}", "Harmonic mean of Precision & Recall"),
                ("Test ROC-AUC", f"{tm['roc_auc']:.4f}", "Ranking quality across thresholds"),
                ("Test PR-AUC", f"{tm['pr_auc']:.4f}", "Precision-Recall area under curve"),
                ("Test Accuracy", f"{tm['accuracy']:.4f}", "Overall classification accuracy"),
            ],
            per_row=6,
        )

        e1, e2 = st.columns(2)
        with e1, ui.chart_card(f"Confusion Matrix — {inspected_algo} (Test Set)", "True Negatives, False Positives, False Negatives, and True Positives"):
            cm = m_detail["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm["matrix"],
                index=[f"Actual: {l}" for l in cm["labels"]],
                columns=[f"Predicted: {l}" for l in cm["labels"]],
            )
            fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Greens", aspect="auto")
            ui.apply_plotly_theme(fig_cm, height=300)
            st.plotly_chart(fig_cm, use_container_width=True)

        with e2, ui.chart_card(f"Classification Report — {inspected_algo} (Test Set)", "Per-class Precision, Recall, F1-Score, and sample support"):
            st.dataframe(pd.DataFrame(m_detail["classification_report"]), use_container_width=True, hide_index=True)

        r1, r2 = st.columns(2)
        with r1, ui.chart_card("ROC Curve Comparison (Held-Out Test Set)", "True Positive Rate vs. False Positive Rate across all trained models"):
            fig_roc = go.Figure()
            colors = ["#2F855A", "#153E2D", "#D69E2E"]
            for idx, (al_name, al_data) in enumerate(train_res["models"].items()):
                c_roc = al_data["curves"]["roc"]
                auc_v = al_data["test_metrics"]["roc_auc"]
                fig_roc.add_scatter(
                    x=c_roc["fpr"],
                    y=c_roc["tpr"],
                    mode="lines",
                    name=f"{al_name} (AUC={auc_v:.3f})",
                    line=dict(color=colors[idx % len(colors)], width=2.5),
                )
            fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline (0.500)", line=dict(color="grey", dash="dash"))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)")
            ui.apply_plotly_theme(fig_roc, height=310)
            st.plotly_chart(fig_roc, use_container_width=True)

        with r2, ui.chart_card("Precision–Recall Curve Comparison (Held-Out Test Set)", "Precision vs. Recall across decision thresholds"):
            fig_pr = go.Figure()
            for idx, (al_name, al_data) in enumerate(train_res["models"].items()):
                c_pr = al_data["curves"]["pr"]
                prauc_v = al_data["test_metrics"]["pr_auc"]
                fig_pr.add_scatter(
                    x=c_pr["recall"],
                    y=c_pr["precision"],
                    mode="lines",
                    name=f"{al_name} (PR-AUC={prauc_v:.3f})",
                    line=dict(color=colors[idx % len(colors)], width=2.5),
                )
            base_rate = ds_info["positive_rate_pct"] / 100.0
            fig_pr.add_scatter(
                x=[0, 1],
                y=[base_rate, base_rate],
                mode="lines",
                name=f"Prevalence ({ds_info['positive_rate_pct']:.1f}%)",
                line=dict(color="grey", dash="dash"),
            )
            fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")
            ui.apply_plotly_theme(fig_pr, height=310)
            st.plotly_chart(fig_pr, use_container_width=True)

        if m_detail.get("feature_importance"):
            with ui.chart_card(f"Top Feature Importances — {inspected_algo}", "Most influential transformed features inside the fitted pipeline"):
                imp_df = pd.DataFrame(m_detail["feature_importance"]).sort_values("importance", ascending=True)
                fig_imp = px.bar(
                    imp_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    text="importance",
                    color_discrete_sequence=["#2F855A"],
                    labels={"importance": "Importance / |Coefficient|", "feature": "Feature"},
                )
                fig_imp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                ui.apply_plotly_theme(fig_imp, height=340)
                st.plotly_chart(fig_imp, use_container_width=True)

# =============================================================================
# TAB 2: PAGEVALUES INVESTIGATION (17 FEATURES VS. 16 FEATURES EXCLUDING PAGEVALUES)
# =============================================================================
with tab_pv:
    with ui.chart_card(
        "PageValues Methodological Investigation (Section 6)",
        "Compare a model trained on all 17 features against the exact same model trained without PageValues (16 features).",
    ):
        pv_c1, pv_c2 = st.columns([2, 1])
        pv_algo = pv_c1.selectbox("Select Algorithm for PageValues Ablation Study", options=classification.ALGORITHMS, index=2)
        run_pv_btn = pv_c2.button("Run PageValues Comparison", icon=":material/compare_arrows:", type="primary", use_container_width=True)

    if run_pv_btn or "pv_study_result" not in st.session_state:
        with st.spinner(f"Training {pv_algo} with all 17 features vs. 16 features (excluding PageValues)..."):
            st.session_state["pv_study_result"] = classification.run_pagevalues_investigation(
                df_raw,
                algorithm_name=pv_algo,
                use_class_weight=True,
                drop_duplicates=bool(st.session_state.get("prep_drop_duplicates", False)),
            )

    pv_res = st.session_state.get("pv_study_result")
    if pv_res:
        pv_df = pd.DataFrame(pv_res["comparison_rows"])
        p1, p2 = st.columns([1.25, 1])
        with p1, ui.chart_card(f"With vs. Without PageValues — {pv_res['algorithm']}", "Evaluated on the identical 80/20 stratified train/test split"):
            st.dataframe(pv_df, use_container_width=True, hide_index=True)

        with p2, ui.chart_card("Visual Metric Shift (17 Features vs. 16 Features)"):
            pv_long = pv_df.melt(
                id_vars=["Metric"],
                value_vars=["With PageValues (17 Features)", "Without PageValues (16 Features)"],
                var_name="Feature Set",
                value_name="Score",
            )
            fig_pv = px.bar(
                pv_long,
                x="Metric",
                y="Score",
                color="Feature Set",
                barmode="group",
                text="Score",
                color_discrete_map={
                    "With PageValues (17 Features)": "#2F855A",
                    "Without PageValues (16 Features)": "#D69E2E",
                },
            )
            fig_pv.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            ui.apply_plotly_theme(fig_pv, height=300)
            st.plotly_chart(fig_pv, use_container_width=True)

        ui.note(
            "<b>Why the Model With the Highest Score Is Not Automatically the Most Appropriate:</b><br>"
            "1. In Google Analytics, <code>PageValues</code> measures the average transaction/goal value assigned to pages "
            "visited prior to an e-commerce conversion. In our dataset inspection, <b>80.61%</b> of purchase sessions have "
            "<code>PageValues &gt; 0</code> (mean <code>27.26</code>) compared to only <b>11.44%</b> of non-purchase sessions (mean <code>1.98</code>).<br>"
            "2. Removing <code>PageValues</code> causes substantial drops in F1, ROC-AUC, and PR-AUC because the model must rely strictly on "
            "page counts, durations, bounce/exit rates, and visitor context.<br>"
            "3. If historical Google Analytics page value attribution is available at inference time, the 17-feature pipeline offers much stronger "
            "discriminative power. However, if a new product page has no historical transaction value attribution yet—or if <code>PageValues</code> "
            "reflects post-checkout goal logging—the 16-feature model provides a more conservative, purely behavioral baseline."
        )

# =============================================================================
# TAB 3: INTERACTIVE PURCHASE PREDICTION
# =============================================================================
with tab_pred:
    saved_models = model_manager.list_saved_classifiers()
    if not saved_models:
        ui.empty_state(
            "No Compatible Trained Model Found",
            "Go to Tab 1 (Model Training, Comparison & Evaluation) and click <b>Train, Cross-Validate & Save Selected Models</b> first.",
            "🛒",
        )
    else:
        with st.container(border=True):
            m_col1, m_col2 = st.columns([1.8, 2.2])
            model_options = {
                m["model_id"]: f"{m['model_name']} ({m['model_id']}) · Test F1={m['test_metrics']['f1']:.3f} · ROC-AUC={m['test_metrics']['roc_auc']:.3f} · {len(m['selected_features'])} features"
                for m in saved_models
            }
            chosen_model_id = m_col1.selectbox(
                "Load Saved Compatible Model Pipeline (Joblib)",
                options=list(model_options.keys()),
                format_func=lambda k: model_options[k],
            )
            loaded_bundle = model_manager.load_classifier_bundle(chosen_model_id) or active_bundle
            m_col2.markdown(
                '<div class="note" style="margin:0">'
                "<b>Prediction Limitation Disclaimer:</b> The UCI dataset contains <b>completed-session summary features</b>. "
                "This prediction estimates a session's purchase likelihood from session-level summaries—it is <b>not</b> validated "
                "for real-time early-session interception and must <b>not</b> be treated as a confirmed transaction."
                "</div>",
                unsafe_allow_html=True,
            )

        req_feats = loaded_bundle["selected_features"]
        defaults = loaded_bundle.get("feature_defaults", {})

        PRESET_SCENARIOS = {
            "Default (Dataset Median / Mode)": defaults,
            "Active Product Shopper (High PageValues & Product Duration)": {
                "Administrative": 3,
                "Administrative_Duration": 95.0,
                "Informational": 1,
                "Informational_Duration": 40.0,
                "ProductRelated": 48,
                "ProductRelated_Duration": 1850.0,
                "BounceRates": 0.004,
                "ExitRates": 0.015,
                "PageValues": 32.5,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            },
            "Quick Bounce Session (1 Product Page, High Bounce/Exit Rate)": {
                "Administrative": 0,
                "Administrative_Duration": 0.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 1,
                "ProductRelated_Duration": 0.0,
                "BounceRates": 0.20,
                "ExitRates": 0.20,
                "PageValues": 0.0,
                "SpecialDay": 0.0,
                "Month": "May",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 1,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            },
            "New Visitor Exploring Products (Moderate Duration, PageValues = 0)": {
                "Administrative": 2,
                "Administrative_Duration": 60.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 25,
                "ProductRelated_Duration": 920.0,
                "BounceRates": 0.005,
                "ExitRates": 0.022,
                "PageValues": 0.0,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "New_Visitor",
                "Weekend": True,
            },
        }

        preset_choice = st.radio(
            "Quick Example Session Profiles (convenience inputs only):",
            options=list(PRESET_SCENARIOS.keys()),
            horizontal=True,
        )
        preset_vals = {**defaults, **(PRESET_SCENARIOS.get(preset_choice) or {})}

        left_form, right_out = st.columns([1.3, 1])
        with left_form:
            with st.form("session_prediction_form", border=True):
                st.markdown('<div class="card-title">Session Feature Inputs</div>', unsafe_allow_html=True)
                st.caption(f"Loaded model requires **{len(req_feats)}** input features (`{loaded_bundle['model_name']}`).")
                input_payload = {}

                f_cols = st.columns(3)
                for idx, feat in enumerate(req_feats):
                    box = f_cols[idx % 3]
                    d_val = preset_vals.get(feat, defaults.get(feat, 0))
                    help_txt = data_validation.FEATURE_DESCRIPTIONS.get(feat, "")

                    if feat == "Month":
                        m_list = data_validation.MONTH_ORDER
                        m_idx = m_list.index(str(d_val)) if str(d_val) in m_list else 0
                        input_payload[feat] = box.selectbox("Month", options=m_list, index=m_idx, help=help_txt)
                    elif feat == "VisitorType":
                        v_list = ["Returning_Visitor", "New_Visitor", "Other"]
                        v_idx = v_list.index(str(d_val)) if str(d_val) in v_list else 0
                        input_payload[feat] = box.selectbox("VisitorType", options=v_list, index=v_idx, help=help_txt)
                    elif feat == "Weekend":
                        input_payload[feat] = bool(box.checkbox("Weekend Session", value=bool(d_val), help=help_txt))
                    elif feat in ("OperatingSystems", "Browser", "Region", "TrafficType"):
                        opts = sorted(df_raw[feat].unique().tolist())
                        o_idx = opts.index(int(d_val)) if int(d_val) in opts else 0
                        input_payload[feat] = int(box.selectbox(feat, options=opts, index=o_idx, help=help_txt))
                    elif feat in ("Administrative", "Informational", "ProductRelated"):
                        input_payload[feat] = int(
                            box.number_input(feat, min_value=0, max_value=1000, value=int(d_val), step=1, help=help_txt)
                        )
                    elif feat in ("BounceRates", "ExitRates"):
                        input_payload[feat] = float(
                            box.number_input(
                                feat, min_value=0.0, max_value=1.0, value=float(d_val), step=0.005, format="%.4f", help=help_txt
                            )
                        )
                    elif feat == "SpecialDay":
                        input_payload[feat] = float(
                            box.slider("SpecialDay (0.0–1.0)", min_value=0.0, max_value=1.0, value=float(d_val), step=0.2, help=help_txt)
                        )
                    else:
                        input_payload[feat] = float(
                            box.number_input(
                                feat, min_value=0.0, max_value=100000.0, value=float(d_val), step=10.0, format="%.2f", help=help_txt
                            )
                        )

                pred_submitted = st.form_submit_button("Predict Session Purchase Intention", icon=":material/bolt:", type="primary", use_container_width=True)

        if pred_submitted or "last_session_prediction" not in st.session_state:
            try:
                st.session_state["last_session_prediction"] = prediction.predict_session(
                    input_payload, model_id=chosen_model_id
                )
            except Exception as exc:
                st.error(str(exc))

        with right_out:
            pred_res = st.session_state.get("last_session_prediction")
            if pred_res:
                is_pos = pred_res["predicted_class"] == 1
                prob_pct = pred_res["purchase_probability"] * 100.0
                st.markdown(
                    f'<div class="pred-banner {"pos" if is_pos else "neg"}">'
                    f'<div class="lab">Model-Based Session Estimate · Not a Confirmed Purchase</div>'
                    f'<div class="big">{"Purchase Intention Detected (Class 1)" if is_pos else "No Purchase Predicted (Class 0)"}</div>'
                    f'<div class="prob">Estimated Purchase Probability: <b>{prob_pct:.2f}%</b> '
                    f'(No-Purchase Probability: <b>{100 - prob_pct:.2f}%</b>)</div>'
                    f'<div style="font-size:0.78rem;opacity:0.9;margin-top:0.5rem">'
                    f'Model: <b>{ui.esc(pred_res["model_name"])}</b> (<code>{ui.esc(pred_res["model_id"])}</code>) · '
                    f'Timestamp: <code>{ui.esc(pred_res["timestamp"])}</code></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                effects = pred_res.get("feature_effects", [])
                if effects:
                    with ui.chart_card(
                        "Local Feature Contribution (Probability Shift vs. Dataset Median/Mode)",
                        "How each session input shifted the predicted purchase probability relative to the baseline session",
                    ):
                        eff_df = pd.DataFrame(effects).sort_values("probability_shift", ascending=True)
                        fig_eff = px.bar(
                            eff_df,
                            x="probability_shift",
                            y="feature",
                            orientation="h",
                            color="probability_shift",
                            color_continuous_scale="RdYlGn",
                            labels={"probability_shift": "Probability Shift", "feature": "Feature"},
                        )
                        ui.apply_plotly_theme(fig_eff, height=270)
                        fig_eff.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_eff, use_container_width=True)

                with st.expander("View Input Summary Sent to Pipeline", icon=":material/data_object:"):
                    st.json(pred_res["inputs"])

# =============================================================================
# TAB 4: LOCAL PREDICTION HISTORY
# =============================================================================
with tab_hist:
    ui.note(
        "<b>Important Distinction:</b> Records below are <b>model-generated purchase intention predictions</b> "
        "stored locally in <code>data/processed/prediction_history.json</code>. They are strictly separate from actual "
        "dataset purchase transactions in <code>online_shoppers_intention.csv</code>."
    )
    hist_items = prediction.load_prediction_history()
    if not hist_items:
        ui.empty_state(
            "No Local Predictions Logged Yet",
            "Run a prediction in Tab 3 (Interactive Purchase Prediction) to log records here.",
            "📜",
        )
    else:
        hist_df = pd.DataFrame(hist_items)
        ui.kpi_row(
            [
                ("Logged Predictions", f"{len(hist_df):,}", "Stored locally in data/processed/"),
                ("Predicted Purchase (Class 1)", f"{int((hist_df['predicted_class'] == 1).sum()):,}", "Estimated positive sessions"),
                ("Mean Purchase Probability", f"{hist_df['purchase_probability_pct'].mean():.2f}%", "Across logged predictions"),
            ],
            per_row=3,
        )

        hb1, hb2 = st.columns(2)
        hb1.download_button(
            f"Export Prediction History ({len(hist_df):,} rows CSV)",
            hist_df.to_csv(index=False).encode("utf-8"),
            file_name="online_shoppers_prediction_history.csv",
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
        )
        if hb2.button("Clear Local Prediction History", icon=":material/delete_outline:", use_container_width=True):
            prediction.clear_prediction_history()
            st.rerun()

        with ui.chart_card("Prediction History Log"):
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
