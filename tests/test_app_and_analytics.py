"""Tests for analytics helpers, React build integration, and Streamlit page rendering."""
import importlib

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from components.streamlit_components import react_components as rc
from config import settings
from src import analytics, feature_engineering as fe, session_engineering as se
from tests.conftest import make_events


def test_analytics_functions(toy_events):
    ev = se.assign_sessions(toy_events, 30)
    k = analytics.compute_kpis(ev)
    assert k["total_events"] == len(ev)
    assert k["unique_visitors"] == 3
    assert k["unique_sessions"] == 4
    assert k["transactions"] == 1
    assert k["purchase_sessions"] == 1

    f_vis = analytics.funnel(ev, "visitors")
    assert list(f_vis["stage"]) == ["view", "addtocart", "transaction"]
    assert f_vis.loc[0, "count"] == 3 and f_vis.loc[2, "count"] == 1

    tr = analytics.event_transitions(ev)
    assert tr.loc["view", "addtocart"] == 1
    assert tr.loc["view", "transaction"] == 1

    ss = fe.build_session_table(ev)
    ab = analytics.cart_abandonment(ss)
    assert ab["sessions_with_cart"] == 1 and ab["cart_and_purchase"] == 1 and ab["abandoned"] == 0

    bp = analytics.interactions_before_purchase(ev)
    assert len(bp) == 1 and bp.loc[0, "events_before_purchase"] == 4

    filt = analytics.filter_events(ev, event_types=["addtocart"])
    assert len(filt) == 1 and (filt["event"] == "addtocart").all()


def test_react_component_bundle_built():
    assert (rc.DIST / "index.html").exists(), "React dist/index.html missing — run npm run build in components/react"
    assert rc.react_available() is True


def test_all_10_page_modules_import():
    assert len(settings.PAGES) == 10
    for key, _, _ in settings.PAGES:
        mod = importlib.import_module(f"app_pages.{key}")
        assert callable(getattr(mod, "render", None)), f"app_pages.{key} missing render()"


@pytest.mark.realdata
def test_streamlit_app_boots_and_renders_pages():
    """Boot the real Streamlit app via AppTest and verify Dashboard & all pages render without exceptions."""
    at = AppTest.from_file(str(settings.PROJECT_ROOT / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Dashboard raised exception: {at.exception}"

    # Verify every page render() function runs cleanly on the real dataset via AppTest.from_string
    for page_key in ("dataset_explorer", "preprocessing", "eda", "customer_segmentation",
                     "model_training", "model_evaluation", "purchase_prediction", "prediction_history", "about"):
        script = (
            "import importlib\n"
            "from components.streamlit_components import ui\n"
            "ui.inject_css()\n"
            f"importlib.import_module('app_pages.{page_key}').render()\n"
        )
        pt = AppTest.from_string(script, default_timeout=60)
        pt.run()
        assert not pt.exception, f"Page {page_key} raised exception: {pt.exception}"
