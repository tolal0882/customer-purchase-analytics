"""Page 10: About Project (CRISP-DM methodology, live experiment results, limitations, references)."""
import pandas as pd
import streamlit as st

from app_pages import common
from components.streamlit_components import react_components as rc, ui
from config import settings
from src import model_manager


def render():
    ui.page_header(
        "About Project",
        "E-commerce Purchase Behavior Prediction and Customer Segmentation Using Machine Learning — CRISP-DM Methodology & Documentation.",
        badges=[
            ("CRISP-DM", "ok"),
            ("Streamlit + React/TS/Tailwind", "info"),
            ("React bundle active" if rc.react_available() else "Native fallback", "ok" if rc.react_available() else "warn"),
        ],
    )

    with ui.chart_card("1 · Project Background & Problem Statement"):
        st.markdown(
            """
            **Background:** E-commerce storefronts generate millions of clickstream interactions (`view`, `addtocart`, `transaction`),
            yet only a small fraction of browsing sessions result in a completed purchase. Understanding *which* active sessions are
            likely to convert—and *how* visitors group by shopping behaviour—enables targeted retention, UX improvements, and conversion
            optimisation without relying on intrusive personal data.

            **Problem Statement:** Raw clickstream logs such as Retailrocket do not provide session identifiers, customer profiles,
            or ready-made prediction labels. Naive models trained on completed sessions suffer from severe **target leakage** (e.g.,
            using post-purchase events or future timestamps to predict an outcome that already occurred). This application implements
            a reproducible, leakage-safe analytical pipeline combining **early-session purchase prediction** and **behavioural segmentation**.
            """
        )

    c1, c2 = st.columns(2)
    with c1, ui.chart_card("2 · Research Objectives"):
        st.markdown(
            """
            1. Validate, clean, and audit the Retailrocket clickstream dataset transparently (without modifying raw files).
            2. Construct reproducible visitor sessions using a configurable inactivity timeout (default: 30 minutes).
            3. Engineer leakage-safe early-session snapshot features and retrospective customer/session behavioural profiles.
            4. Train and compare four classification algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
               using chronological evaluation and imbalanced-class metrics.
            5. Segment customers and sessions via K-Means with Elbow, Silhouette, and 2-D PCA inspection, generating data-driven profiles.
            6. Deliver an interactive hybrid Streamlit + React/TypeScript/Tailwind web application for exploration, training, and inference.
            """
        )
    with c2, ui.chart_card("3 · Research Questions"):
        st.markdown(
            """
            - **RQ1:** How accurately can early-session interaction signals (observed after the first *N* non-transaction events or *T* minutes)
              predict whether a transaction will occur later in the same session?
            - **RQ2:** How do linear, single-tree, bagging, and boosting classifiers compare under chronological train/test evaluation
              on imbalanced purchase outcomes (PR-AUC, ROC-AUC, Recall, Precision, F1)?
            - **RQ3:** What distinct behavioural clusters emerge among active visitors and multi-event sessions when grouped using
              scaled, skew-adjusted interaction features, and how do their post-clustering purchase rates differ?
            """
        )

    with ui.chart_card("4 · CRISP-DM Methodology"):
        crisp = pd.DataFrame([
            {"Phase": "1. Business Understanding",
             "Implementation in this Application": "Defined dual objective: early-session purchase prediction + behavioural customer/session segmentation with strict leakage controls."},
            {"Phase": "2. Data Understanding",
             "Implementation in this Application": "Dataset Explorer & EDA pages inspect raw schemas, missing values, duplicates, timestamps, funnels, transitions, products, and categories."},
            {"Phase": "3. Data Preparation",
             "Implementation in this Application": "Logged cleaning (non-silent duplicate/invalid handling), gap-based session construction (30-min default), as-of category enrichment, and snapshot generation."},
            {"Phase": "4. Modeling",
             "Implementation in this Application": "Interactive Scikit-learn Pipelines (ColumnTransformer + imputation + scaling + optional training-fold sampling) for 4 classifiers and K-Means clustering."},
            {"Phase": "5. Evaluation",
             "Implementation in this Application": "Chronological train/test split, expanding-window TimeSeriesSplit CV, PR-AUC/ROC-AUC/F1/Confusion Matrix, Elbow & Silhouette analysis."},
            {"Phase": "6. Deployment",
             "Implementation in this Application": "Hybrid Streamlit + React/TypeScript/Tailwind dashboard with joblib model persistence, interactive inference, and local prediction history."},
        ])
        st.dataframe(crisp, use_container_width=True, hide_index=True)

    core = common.get_core()
    with ui.chart_card("5 · Dataset Description, Preparation & Session Construction"):
        if core is not None:
            m = core.meta
            q = core.quality
            st.markdown(
                f"""
                - **Source:** Retailrocket Recommender System Dataset (Kaggle).
                - **Raw `events.csv`:** `{q['rows']:,}` rows spanning `{q.get('timestamp_min', '—')[:10]}` to `{q.get('timestamp_max', '—')[:10]}`
                  (`{q.get('span_days', 0):.0f}` days), `{q.get('unique_visitorid', 0):,}` unique visitors, `{q.get('unique_itemid', 0):,}` unique items.
                - **Active processed version:** `{m['dataset_version']}` — `{m['rows_clean']:,}` cleaned events,
                  `{m['n_sessions']:,}` engineered sessions (timeout: `{m['session_timeout_minutes']:g}` min),
                  `{m['n_visitors']:,}` visitors, categories attached: `{m['categories_attached']}`.
                - **Session Construction:** Events are stably sorted by `(visitorid, timestamp)`. A new `session_id` starts whenever a visitor's
                  inter-event gap exceeds the configured timeout (default **30 minutes**, an analytical assumption not present in the raw file).
                - **Feature Engineering:** Session-level features (`total_events`, `total_views`, `unique_items_viewed`, `add_to_cart_count`,
                  `session_duration_sec`, `avg_time_between_events_sec`, `cart_interaction_ratio`, `item_revisit_ratio`, `first_cart_position`,
                  `ends_with_cart`, `start_hour`, `start_dayofweek`, optional as-of `unique_categories_viewed`) and historical visitor features
                  (strictly earlier sessions only).
                """
            )
        else:
            st.info("Load `events.csv` on Data Preprocessing to populate live dataset statistics here.")

    with ui.chart_card("6 · Machine Learning Methodology & Leakage Prevention"):
        st.markdown(
            """
            - **Early-Session Prediction Snapshot:** Rather than using completed-session aggregates to predict that same session,
              a snapshot is taken after the **3rd non-transaction event** (or a configurable time cutoff). Only events at or before the snapshot
              enter the feature vector; the label `target = 1` indicates a `transaction` strictly *after* the snapshot within the same session.
            - **Strict Leakage Safeguards:**
              1. Sessions where a transaction occurs at or before the snapshot are excluded.
              2. `transaction_count`, `visitorid`, `itemid`, and `transactionid` are explicitly blocked from model inputs.
              3. Chronological train/test split and expanding-window `TimeSeriesSplit` CV prevent future-to-past leakage.
              4. Imputation, scaling, and optional resampling live inside a single Scikit-learn pipeline fitted on training folds only.
            - **Evaluation Metrics:** Because conversion is imbalanced (~7.5% of eligible snapshots), models are evaluated with
              **PR-AUC**, **ROC-AUC**, **Precision**, **Recall**, **F1-score**, **Confusion Matrix**, and **Accuracy** (compared against the majority baseline).
            - **Segmentation:** K-Means on scaled, `log1p`-transformed behavioural features (excluding `visitorid`). Cluster names are
              generated programmatically from feature z-scores; purchase rates are compared *after* clustering.
            """
        )

    with ui.chart_card("7 · Actual Experiment Results (Live from experiments/)"):
        exps = model_manager.list_experiments()
        if not exps:
            st.info("No saved classification experiments yet. Train models on the **Model Training** page to record real results here.")
        else:
            tbl = model_manager.experiments_table(exps)
            st.dataframe(tbl.round(4), use_container_width=True, hide_index=True)

    with ui.chart_card("8 · Limitations & Methodological Caveats"):
        st.markdown(
            """
            1. **No `removefromcart` event:** Retailrocket only logs `view`, `addtocart`, and `transaction`. Therefore `remove_from_cart_count`
               cannot be computed, and `historical_cart_abandonment_rate` is a proxy (cart sessions without a purchase in that same session).
            2. **Analytical session timeout:** The 30-minute inactivity boundary is an assumption; sensitivity analysis on the Preprocessing page
               shows how session counts vary across timeouts.
            3. **Engaged-session scope:** ~78% of sessions contain only 1 event. Early-session prediction after 3 events applies specifically to
               engaged multi-event sessions (~95.6k eligible samples), not single-click bounces.
            4. **Partial category coverage:** Only ~78.8% of event items appear in `item_properties_part1/2.csv`, and an as-of join leaves
               events prior to an item's first property timestamp with unknown category.
            5. **Behavioural clusters ≠ demographics:** K-Means segments reflect observed clickstream patterns only, not verified customer personas.
            6. **Temporal drift:** The final month (September 2015) is partial and forms the chronological test window; seasonal shifts may affect generalisation.
            """
        )

    with ui.chart_card("9 · References"):
        st.markdown(
            """
            - Retailrocket Recommender System Dataset (Kaggle): https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
            - Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide.*
            - Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
            - Streamlit Documentation: https://docs.streamlit.io
            - Plotly Python Graphing Library: https://plotly.com/python/
            """
        )
