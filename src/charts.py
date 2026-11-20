"""Plotly figure builders (pure functions; consistent theme)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import settings

C = settings.COLORS
PALETTE = settings.CHART_COLORS
EVENT_COLORS = {"view": "#2F855A", "addtocart": "#D69E2E", "transaction": "#153E2D"}


def theme(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=C["text"], size=12), colorway=PALETTE,
        showlegend=legend, legend=dict(orientation="h", y=-0.2), hoverlabel=dict(font_size=12))
    fig.update_xaxes(gridcolor="#E6EEE8", zeroline=False)
    fig.update_yaxes(gridcolor="#E6EEE8", zeroline=False)
    return fig


def bar(df, x, y, color=None, orientation="v", height=340, color_map=None, text_auto=True, log_y=False, **kw):
    fig = px.bar(df, x=x, y=y, color=color, orientation=orientation, text_auto=text_auto and ".3s",
                 color_discrete_map=color_map or EVENT_COLORS, log_y=log_y, **kw)
    return theme(fig, height, legend=color is not None)


def line(df, x, y, color=None, height=340, **kw):
    fig = px.line(df, x=x, y=y, color=color, color_discrete_map=EVENT_COLORS, **kw)
    return theme(fig, height, legend=color is not None)


def funnel_fig(f: pd.DataFrame, height=320):
    fig = go.Figure(go.Funnel(y=f["stage"], x=f["count"], textinfo="value+percent initial",
                              marker=dict(color=[EVENT_COLORS[s] for s in f["stage"]])))
    return theme(fig, height, legend=False)


def hist_bar(h: pd.DataFrame, xlabel: str, height=320, log_y=True):
    fig = px.bar(h, x="bin", y="count", log_y=log_y, labels={"bin": xlabel}, color_discrete_sequence=[PALETTE[0]])
    return theme(fig, height, legend=False)


def heatmap(m: pd.DataFrame, title="", height=320, fmt=",d", pct=False):
    z = m.values.astype(float)
    if pct:
        z = z / np.clip(z.sum(axis=1, keepdims=True), 1, None) * 100
    fig = go.Figure(go.Heatmap(z=z, x=list(m.columns), y=list(m.index), colorscale=[[0, "#F3F7F3"], [1, C["forest"]]],
                               text=np.round(z, 1) if pct else m.values, texttemplate="%{text}", hoverongaps=False))
    fig.update_layout(title=title, xaxis_title="Next event", yaxis_title="Current event")
    return theme(fig, height, legend=False)


def confusion_fig(cm, height=340):
    cm = np.array(cm)
    labels = ["No purchase", "Purchase"]
    pct = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100
    text = [[f"{cm[i, j]:,}<br>{pct[i, j]:.1f}% of actual" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=pct, x=labels, y=labels, text=text, texttemplate="%{text}",
                               colorscale=[[0, "#F3F7F3"], [1, C["primary"]]], showscale=False))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", yaxis_autorange="reversed")
    return theme(fig, height, legend=False)


def roc_fig(curves: dict[str, dict], height=360):
    fig = go.Figure()
    for i, (name, c) in enumerate(curves.items()):
        fig.add_scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=name, line=dict(color=PALETTE[i % len(PALETTE)], width=2))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="#999"))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    return theme(fig, height)


def pr_fig(curves: dict[str, dict], prevalence: float | None = None, height=360):
    fig = go.Figure()
    for i, (name, c) in enumerate(curves.items()):
        fig.add_scatter(x=c["recall"], y=c["precision"], mode="lines", name=name,
                        line=dict(color=PALETTE[i % len(PALETTE)], width=2))
    if prevalence is not None:
        fig.add_scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines", name="No-skill (prevalence)",
                        line=dict(dash="dash", color="#999"))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 1.02])
    return theme(fig, height)


def importance_fig(imp: dict[str, float], title="", height=360, signed=False):
    s = pd.Series(imp).sort_values(key=lambda x: x.abs())
    colors = [C["primary"] if v >= 0 else C["danger"] for v in s.values] if signed else C["primary"]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors))
    fig.update_layout(title=title, title_font_size=13)
    return theme(fig, height, legend=False)


def model_compare_fig(table: pd.DataFrame, metrics: list[str], label_col="label", height=380):
    long = table.melt(id_vars=[label_col], value_vars=metrics, var_name="metric", value_name="value")
    fig = px.bar(long, x="metric", y="value", color=label_col, barmode="group", color_discrete_sequence=PALETTE,
                 text_auto=".3f")
    fig.update_layout(yaxis_range=[0, 1.05])
    return theme(fig, height)


def elbow_fig(ka: pd.DataFrame, height=320):
    fig = go.Figure(go.Scatter(x=ka["k"], y=ka["inertia"], mode="lines+markers", line=dict(color=C["primary"])))
    fig.update_layout(xaxis_title="K", yaxis_title="Inertia (within-cluster SSE)", xaxis=dict(dtick=1))
    return theme(fig, height, legend=False)


def silhouette_fig(ka: pd.DataFrame, height=320):
    fig = go.Figure(go.Bar(x=ka["k"], y=ka["silhouette"], marker_color=C["accent"], text=ka["silhouette"].round(3),
                           textposition="outside"))
    fig.update_layout(xaxis_title="K", yaxis_title="Silhouette score (sampled)", xaxis=dict(dtick=1))
    return theme(fig, height, legend=False)


def pca_fig(coords: pd.DataFrame, names: dict, explained: list, height=420):
    d = coords.assign(segment=coords["cluster"].map(lambda c: f"{c}: {names.get(c, c)}"))
    fig = px.scatter(d, x="PC1", y="PC2", color="segment", opacity=0.55, color_discrete_sequence=PALETTE)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(xaxis_title=f"PC1 ({explained[0] * 100:.1f}% var.)", yaxis_title=f"PC2 ({explained[1] * 100:.1f}% var.)")
    return theme(fig, height)


def cluster_size_fig(profile: pd.DataFrame, names: dict, height=320):
    d = profile.reset_index()
    d["label"] = d["cluster"].map(lambda c: f"{c}: {names.get(c, c)}")
    fig = px.bar(d, x="label", y="size", text=d["share_pct"].map(lambda v: f"{v:.1f}%"), color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(xaxis_title="", yaxis_title="Members")
    return theme(fig, height, legend=False)


def z_heatmap(z: pd.DataFrame, names: dict, height=360):
    fig = go.Figure(go.Heatmap(z=z.values, x=list(z.columns), y=[f"{c}: {names.get(c, c)}" for c in z.index],
                               colorscale="RdYlGn", zmid=0, text=z.round(2).values, texttemplate="%{text}"))
    return theme(fig, height, legend=False)
