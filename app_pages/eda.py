"""Page 4: Exploratory Data Analysis (all charts computed from the processed data)."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app_pages import common
from components.streamlit_components import ui
from config import settings
from src import analytics, charts


# Heavy computations are cached per dataset version; arguments are small so hashing is cheap.
@st.cache_data(show_spinner=False)
def _transitions(v):
    return analytics.event_transitions(common.get_core().events)


@st.cache_data(show_spinner=False)
def _daily(v):
    return analytics.time_series(common.get_core().events, "D")


@st.cache_data(show_spinner=False)
def _visitor_activity(v):
    return analytics.visitor_activity_distribution(common.get_core().events)


@st.cache_data(show_spinner=False)
def _before_purchase(v):
    return analytics.interactions_before_purchase(common.get_core().events)


@st.cache_data(show_spinner=False)
def _top_items(v, by):
    return analytics.top_items(common.get_core().events, 20, by)


@st.cache_data(show_spinner=False)
def _item_conv(v, min_carts):
    return analytics.item_conversion(common.get_core().events, min_carts, 20)


@st.cache_data(show_spinner=False)
def _cat_perf(v):
    core = common.get_core()
    return analytics.category_performance(core.events, common.get_category_tree(), 25)


@st.cache_data(show_spinner=False)
def _cat_event_counts(v):
    ev = common.get_core().events
    sub = ev[ev["categoryid"].notna()]
    return sub.groupby("categoryid", observed=True).size().rename("events").reset_index().astype({"categoryid": "int64"})


def render():
    ui.page_header("Exploratory Data Analysis", "Event, visitor, session, purchase, product and category analysis on the full processed dataset.")
    core = common.require_core()
    v = core.meta["dataset_version"]
    ss = core.sessions
    tabs = st.tabs(["A · Events", "B · Visitors", "C · Sessions", "D · Purchases", "E · Products", "F · Categories"])

    with tabs[0]:
        a, b = st.columns(2)
        with a, ui.chart_card("Event distribution"):
            st.plotly_chart(charts.bar(analytics.event_distribution(core.events), "event", "count", color="event"), use_container_width=True)
        with b, ui.chart_card("Event frequency over time (daily)"):
            d = _daily(v)
            ev_sel = st.multiselect("Events", settings.VALID_EVENTS, default=["addtocart", "transaction"], key="eda_ev_sel")
            st.plotly_chart(charts.line(d[d["event"].isin(ev_sel)], "period", "count", color="event"), use_container_width=True)
            st.caption("Views are ~38× more frequent than carts; select 'view' to include them.")
        with ui.chart_card("Event transitions (next event within the same session)", "Rows = current event, columns = the event that follows"):
            t = _transitions(v)
            norm = st.checkbox("Row-normalise (% of transitions from each event)", True, key="eda_norm")
            st.plotly_chart(charts.heatmap(t, pct=norm), use_container_width=True)
            st.caption("The last event of a session has no successor and is excluded.")

    with tabs[1]:
        act = _visitor_activity(v)
        nsess = ss.groupby("visitorid").size()
        ui.kpi_row([
            ("Unique visitors", common.fmt_int(len(act)), ""),
            ("Returning visitors (≥2 sessions)", common.fmt_int((nsess > 1).sum()), f"{(nsess > 1).mean() * 100:.1f}% of visitors"),
            ("Median events / visitor", f"{act.median():.0f}", f"mean {act.mean():.2f}"),
            ("Max events / visitor", common.fmt_int(act.max()), "possible bot / heavy user"),
        ])
        a, b = st.columns(2)
        with a, ui.chart_card("Events per visitor"):
            h = analytics.hist_counts(act, [0, 1, 2, 3, 5, 10, 20, 50, 100, 1e9])
            h["bin"] = ["1", "2", "3", "4–5", "6–10", "11–20", "21–50", "51–100", ">100"]
            st.plotly_chart(charts.hist_bar(h, "Events per visitor"), use_container_width=True)
        with b, ui.chart_card("Sessions per visitor (returning activity)"):
            h = analytics.hist_counts(nsess, [0, 1, 2, 3, 5, 10, 1e9])
            h["bin"] = ["1", "2", "3", "4–5", "6–10", ">10"]
            st.plotly_chart(charts.hist_bar(h, "Sessions per visitor"), use_container_width=True)
        with ui.chart_card("Most active visitors"):
            top = act.sort_values(ascending=False).head(15).rename("events").reset_index()
            top["visitorid"] = top["visitorid"].astype(str)
            st.plotly_chart(charts.bar(top, "visitorid", "events", height=300), use_container_width=True)

    with tabs[2]:
        multi = ss[ss["total_events"] >= 2]
        ui.kpi_row([
            ("Sessions", common.fmt_int(len(ss)), ""), ("Multi-event sessions", common.fmt_int(len(multi)), f"{len(multi) / len(ss) * 100:.1f}%"),
            ("Median duration (multi-event)", f"{multi['session_duration_sec'].median():.0f} s", ""),
            ("Mean views / session", f"{ss['total_views'].mean():.2f}", ""),
        ])
        a, b = st.columns(2)
        with a, ui.chart_card("Session duration (sessions with ≥2 events)"):
            h = analytics.hist_counts(multi["session_duration_sec"], [-1, 10, 30, 60, 300, 600, 1200, 1800, 1e9])
            h["bin"] = ["≤10s", "10–30s", "30–60s", "1–5m", "5–10m", "10–20m", "20–30m", ">30m"]
            st.plotly_chart(charts.hist_bar(h, "Duration"), use_container_width=True)
        with b, ui.chart_card("Events per session"):
            h = analytics.hist_counts(ss["total_events"], [0, 1, 2, 3, 5, 10, 20, 50, 1e9])
            h["bin"] = ["1", "2", "3", "4–5", "6–10", "11–20", "21–50", ">50"]
            st.plotly_chart(charts.hist_bar(h, "Events per session"), use_container_width=True)
        a, b = st.columns(2)
        with a, ui.chart_card("Views per session"):
            h = analytics.hist_counts(ss["total_views"], [-1, 0, 1, 2, 3, 5, 10, 20, 1e9])
            h["bin"] = ["0", "1", "2", "3", "4–5", "6–10", "11–20", ">20"]
            st.plotly_chart(charts.hist_bar(h, "Views per session"), use_container_width=True)
        with b, ui.chart_card("Session start pattern (hour × weekday)"):
            pv = ss.groupby(["start_dayofweek", "start_hour"]).size().unstack(fill_value=0).reindex(range(7), fill_value=0)
            pv.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig = px.imshow(pv, aspect="auto", color_continuous_scale=["#F3F7F3", settings.COLORS["forest"]], labels=dict(x="Hour (UTC)", y="", color="Sessions"))
            st.plotly_chart(charts.theme(fig, 320, False), use_container_width=True)

    with tabs[3]:
        ab = analytics.cart_abandonment(ss)
        ui.kpi_row([
            ("Sessions with add-to-cart", common.fmt_int(ab["sessions_with_cart"]), ""),
            ("Cart → purchase", common.fmt_int(ab["cart_and_purchase"]), f"{(100 - ab['abandonment_rate']):.1f}% of cart sessions"),
            ("Cart abandonment", f"{ab['abandonment_rate']:.1f}%", f"{common.fmt_int(ab['abandoned'])} cart sessions without purchase"),
            ("Purchases without cart event", common.fmt_int(ab["purchase_without_cart"]), "sessions with transaction, no add-to-cart"),
        ])
        st.caption("Abandonment is a proxy: the dataset has no remove-from-cart event, and a cart may be bought in a later session.")
        a, b = st.columns(2)
        with a, ui.chart_card("Transaction trend (daily)"):
            d = _daily(v)
            st.plotly_chart(charts.line(d[d["event"] == "transaction"], "period", "count", color=None), use_container_width=True)
        with b, ui.chart_card("Add-to-cart → transaction conversion (sessions)"):
            f = pd.DataFrame({"stage": ["Sessions with add-to-cart", "…that also purchased"],
                              "count": [ab["sessions_with_cart"], ab["cart_and_purchase"]]})
            st.plotly_chart(charts.bar(f, "stage", "count", color=None, height=320), use_container_width=True)
        with ui.chart_card("Product interaction before purchase", "Events (and views) preceding the first transaction in purchase sessions"):
            bp = _before_purchase(v)
            if bp.empty:
                ui.empty_state("No purchase sessions", "", "🧾")
            else:
                st.caption(f"{len(bp):,} purchase sessions · median {bp['events_before_purchase'].median():.0f} events and {bp['views_before_purchase'].median():.0f} views before purchase")
                h = analytics.hist_counts(bp["events_before_purchase"], [-1, 0, 1, 2, 3, 5, 10, 20, 1e9])
                h["bin"] = ["0", "1", "2", "3", "4–5", "6–10", "11–20", ">20"]
                st.plotly_chart(charts.hist_bar(h, "Events before first purchase", log_y=False), use_container_width=True)

    with tabs[4]:
        by = st.radio("Rank products by", ["view", "addtocart", "transaction"], horizontal=True, key="eda_prod_by")
        ti = _top_items(v, by)
        ti["item"] = ti["itemid"].astype(str)
        a, b = st.columns(2)
        with a, ui.chart_card(f"Most popular products by {by}"):
            st.plotly_chart(charts.bar(ti.melt("item", ["view", "addtocart", "transaction"], var_name="event", value_name="count"),
                                       "count", "item", color="event", orientation="h", text_auto=False, height=520), use_container_width=True)
        with b, ui.chart_card("Best add-to-cart → transaction ratio", "Products with at least N add-to-cart events"):
            mc = st.slider("Minimum add-to-cart events", 3, 50, 10, key="eda_minc")
            ic = _item_conv(v, mc)
            ic["item"] = ic["itemid"].astype(str)
            st.plotly_chart(charts.bar(ic, "transactions_per_100_carts", "item", orientation="h", color=None, height=520), use_container_width=True)
            st.caption("Ratio can exceed 100: items are often bought without a prior add-to-cart event or carts are repeated.")
        with st.expander("Table: top products"):
            st.dataframe(ti.drop(columns="item"), use_container_width=True, hide_index=True)

    with tabs[5]:
        tree = common.get_category_tree()
        has_cat = "categoryid" in core.events.columns and core.events["categoryid"].notna().any()
        if tree is None:
            ui.empty_state("category_tree.csv not found", data_loader_hint(), "🌳")
        else:
            roots = analytics.category_roots(tree)
            st.caption(f"{len(tree):,} categories · {int(tree['parentid'].isna().sum())} roots · max depth {int(roots['depth'].max())}")
        if tree is not None and not has_cat:
            ui.note("Category performance needs item categories from the optional <b>item_properties</b> files. "
                    "Enable <i>Attach item categories</i> on the <b>Data Preprocessing</b> page (one-time chunked read, cached).")
            with ui.chart_card("Category tree structure (from category_tree.csv)"):
                cnt = roots.groupby("root").size().rename("categories").reset_index()
                cnt["root"] = cnt["root"].astype(str)
                st.plotly_chart(charts.bar(cnt.sort_values("categories", ascending=False).head(25), "root", "categories", color=None), use_container_width=True)
                st.caption("Number of categories under each root category.")
        elif has_cat:
            cov = core.events["categoryid"].notna().mean() * 100
            st.caption(f"Category known for {cov:.1f}% of events (as-of join: earlier events of items whose first property record is later are unknown).")
            cp = _cat_perf(v)
            cp["category"] = cp["categoryid"].astype(str)
            a, b = st.columns(2)
            with a, ui.chart_card("Category interactions (top 25)"):
                st.plotly_chart(charts.bar(cp.melt("category", ["view", "addtocart", "transaction"], var_name="event", value_name="count"),
                                           "count", "category", color="event", orientation="h", text_auto=False, height=560), use_container_width=True)
            with b, ui.chart_card("Category performance: transactions per 100 add-to-carts"):
                st.plotly_chart(charts.bar(cp.dropna(subset=["transactions_per_100_carts"]), "transactions_per_100_carts", "category", orientation="h", color=None, height=560), use_container_width=True)
            if tree is not None:
                with ui.chart_card("Category tree (root → category), sized by events"):
                    ec = _cat_event_counts(v).merge(roots, on="categoryid", how="left")
                    ec = ec.dropna(subset=["root"]).sort_values("events", ascending=False).head(300)
                    ec["root"] = "Root " + ec["root"].astype(int).astype(str)
                    ec["cat"] = ec["categoryid"].astype(str)
                    fig = px.treemap(ec, path=["root", "cat"], values="events", color_discrete_sequence=settings.CHART_COLORS)
                    st.plotly_chart(charts.theme(fig, 520, False), use_container_width=True)
                    st.caption("Top 300 categories by events.")


def data_loader_hint():
    return "Place category_tree.csv in the raw data folder (see Data Preprocessing)."
