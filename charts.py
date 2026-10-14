"""
Altair charts for the RAGSmith UI.

All charts are single-series, so color only encodes magnitude (one blue
ramp) and never identity -- no legends needed. Marks are thin with rounded
data ends, gridlines stay recessive, and every chart has a hover tooltip.
Light and dark mode get their own steps of the same ramp.
"""

import altair as alt
import pandas as pd
import streamlit as st

PALETTE = {
    "light": {
        "series": "#2a78d6",
        "ramp": ["#86b6ef", "#104281"],
        "surface": "#f9f9f7",
        "grid": "#e1e0d9",
        "muted": "#898781",
    },
    "dark": {
        "series": "#3987e5",
        "ramp": ["#2a78d6", "#cde2fb"],
        "surface": "#111110",
        "grid": "#2c2c2a",
        "muted": "#898781",
    },
}


def _colors() -> dict:
    try:
        mode = st.context.theme.type or "light"
    except Exception:
        mode = "light"
    return PALETTE.get(mode, PALETTE["light"])


def _finish(chart: alt.Chart, height: int) -> alt.Chart:
    c = _colors()
    return (
        chart.properties(height=height)
        .configure_axis(
            gridColor=c["grid"], domainColor=c["grid"], tickColor=c["grid"],
            labelColor=c["muted"], titleColor=c["muted"], labelFontSize=11, titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_view(strokeWidth=0)
    )


def page_density(chars_per_page: list[int]) -> alt.Chart:
    """Extracted characters per page; near-empty pages usually need OCR."""
    c = _colors()
    df = pd.DataFrame({"page": range(1, len(chars_per_page) + 1), "characters": chars_per_page})
    tooltip = [alt.Tooltip("page:Q", title="Page"), alt.Tooltip("characters:Q", title="Characters", format=",")]
    y = alt.Y("characters:Q", title="Characters")
    if len(df) <= 120:
        x = alt.X("page:O", title="Page", scale=alt.Scale(paddingInner=0.2),
                  axis=alt.Axis(labelAngle=0, labelOverlap=True))
        mark = alt.Chart(df).mark_bar(color=c["series"], cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                                      **({"size": 40} if len(df) <= 8 else {}))
    else:
        x = alt.X("page:Q", title="Page", scale=alt.Scale(nice=False), axis=alt.Axis(format="d"))
        mark = alt.Chart(df).mark_area(color=c["series"], opacity=0.25,
                                       line={"color": c["series"], "strokeWidth": 2})
    mark = mark.encode(x=x, y=y, tooltip=tooltip)
    return _finish(mark, 220)


def size_histogram(lengths: list[int]) -> alt.Chart:
    c = _colors()
    df = pd.DataFrame({"chars": lengths})
    chart = (
        alt.Chart(df)
        .mark_bar(color=c["series"], cornerRadiusTopLeft=4, cornerRadiusTopRight=4, binSpacing=2)
        .encode(
            x=alt.X("chars:Q", bin=alt.Bin(maxbins=30), title="Chunk size (characters)"),
            y=alt.Y("count():Q", title="Chunks", axis=alt.Axis(tickMinStep=1, format="d")),
            tooltip=[
                alt.Tooltip("chars:Q", bin=alt.Bin(maxbins=30), title="Size range"),
                alt.Tooltip("count():Q", title="Chunks"),
            ],
        )
    )
    return _finish(chart, 230)


def projection(df: pd.DataFrame) -> alt.Chart:
    """Scatter of 2-D projected chunk vectors, shaded by page (position in the document)."""
    c = _colors()
    chart = (
        alt.Chart(df)
        .mark_circle(size=70, opacity=0.9, stroke=c["surface"], strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", title="Component 1", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("y:Q", title="Component 2", axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("page:Q", title="Page", scale=alt.Scale(range=c["ramp"]),
                            legend=alt.Legend(orient="right", gradientLength=140, titleFontWeight="normal",
                                              format="d", tickMinStep=1)),
            tooltip=[
                alt.Tooltip("chunk:Q", title="Chunk"),
                alt.Tooltip("page:Q", title="Page"),
                alt.Tooltip("preview:N", title="Text"),
            ],
        )
        .interactive()
    )
    return _finish(chart, 360)


def score_profile(df: pd.DataFrame) -> alt.Chart:
    """Horizontal bars of retrieval score by rank."""
    c = _colors()
    chart = (
        alt.Chart(df)
        .mark_bar(color=c["series"], cornerRadiusEnd=4, height={"band": 0.75})
        .encode(
            y=alt.Y("rank:O", title="Rank"),
            x=alt.X("score:Q", title="Score"),
            tooltip=[
                alt.Tooltip("rank:O", title="Rank"),
                alt.Tooltip("chunk:Q", title="Chunk"),
                alt.Tooltip("page:Q", title="Page"),
                alt.Tooltip("score:Q", title="Score", format=".3f"),
            ],
        )
    )
    return _finish(chart, max(140, 26 * len(df)))
