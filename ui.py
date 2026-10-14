"""
Presentation helpers for the RAGSmith UI: global styles, the pipeline
stepper, stage headers and small HTML components Streamlit has no native
widget for. Colors are derived from `currentColor` so every component
follows Streamlit's light/dark theme without knowing which one is active.
"""

from html import escape

import streamlit as st

CSS = """
<style>
:root {
  --rs-accent: #2a78d6;
  --rs-line: color-mix(in srgb, currentColor 12%, transparent);
  --rs-wash: color-mix(in srgb, currentColor 4%, transparent);
  --rs-muted: color-mix(in srgb, currentColor 58%, transparent);
  --rs-faint: color-mix(in srgb, currentColor 38%, transparent);
  --rs-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

.block-container { padding-top: 2.2rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }

/* ---- page header ---- */
.rs-eyebrow {
  font-family: var(--rs-mono); font-size: 11px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--rs-muted); margin-bottom: 4px;
}
.rs-title { font-size: 26px; font-weight: 600; letter-spacing: -.01em; margin: 0; line-height: 1.25; }
.rs-subtitle { color: var(--rs-muted); font-size: 14px; margin-top: 4px; }

/* ---- stepper ---- */
.rs-stepper { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin: 22px 0 8px; }
.rs-step { border-top: 2px solid var(--rs-line); padding-top: 10px; min-width: 0; }
.rs-step.is-done { border-top-color: var(--rs-accent); }
.rs-step.is-active { border-top-color: var(--rs-accent); }
.rs-step.is-active .rs-step-name { color: inherit; }
.rs-step-head { display: flex; align-items: baseline; gap: 8px; }
.rs-step-index { font-family: var(--rs-mono); font-size: 11px; color: var(--rs-faint); }
.rs-step-name { font-weight: 600; font-size: 13px; color: var(--rs-muted); }
.rs-step.is-done .rs-step-name { color: inherit; }
.rs-step-state {
  font-family: var(--rs-mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--rs-faint); margin-top: 2px;
}
.rs-step.is-active .rs-step-state, .rs-step.is-done .rs-step-state { color: var(--rs-accent); }
.rs-step-meta {
  font-size: 12px; color: var(--rs-muted); margin-top: 4px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ---- stage panels ---- */
.rs-panel-head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.rs-panel-index { font-family: var(--rs-mono); font-size: 12px; color: var(--rs-faint); }
.rs-panel-title { font-size: 17px; font-weight: 600; }
.rs-panel-desc { color: var(--rs-muted); font-size: 13px; margin: 2px 0 6px; }
.rs-panel-summary {
  font-family: var(--rs-mono); font-size: 12px; color: var(--rs-muted);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.rs-locked { opacity: .55; }

.rs-pill {
  font-family: var(--rs-mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
  padding: 2px 8px; border-radius: 999px; border: 1px solid var(--rs-line); color: var(--rs-muted);
}
.rs-pill.is-active { color: var(--rs-accent); border-color: color-mix(in srgb, var(--rs-accent) 45%, transparent); }
.rs-pill.is-done { color: var(--rs-accent); background: color-mix(in srgb, var(--rs-accent) 10%, transparent); border-color: transparent; }

.rs-section {
  font-family: var(--rs-mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--rs-muted); border-bottom: 1px solid var(--rs-line); padding-bottom: 6px; margin: 6px 0 10px;
}

/* ---- stat tiles ---- */
.rs-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-bottom: 10px; }
.rs-stat { border: 1px solid var(--rs-line); border-radius: 6px; padding: 10px 12px; background: var(--rs-wash); }
.rs-stat-label {
  font-family: var(--rs-mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: var(--rs-muted);
}
.rs-stat-value { font-size: 20px; font-weight: 600; margin-top: 2px; letter-spacing: -.01em; }
.rs-stat-unit { font-size: 12px; font-weight: 400; color: var(--rs-muted); margin-left: 3px; }

/* ---- key/value spec sheet (sidebar) ---- */
.rs-spec { margin: 0; font-size: 13px; }
.rs-spec dt {
  font-family: var(--rs-mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--rs-muted); margin-top: 12px;
}
.rs-spec dt:first-child { margin-top: 0; }
.rs-spec dd { margin: 2px 0 0; font-weight: 500; overflow-wrap: break-word; }
.rs-spec dd small {
  display: block; font-family: var(--rs-mono); font-size: 11px; font-weight: 400;
  color: var(--rs-muted); margin-top: 1px; overflow-wrap: anywhere;
}
.rs-spec dd.is-empty { color: var(--rs-faint); font-weight: 400; }
.rs-brand { font-weight: 700; font-size: 18px; letter-spacing: -.01em; }
.rs-brand-sub { color: var(--rs-muted); font-size: 12px; margin-bottom: 14px; }

/* ---- source citations ---- */
.rs-source { border-left: 2px solid var(--rs-line); padding: 2px 0 2px 12px; margin: 10px 0; }
.rs-source-meta { font-family: var(--rs-mono); font-size: 11px; color: var(--rs-muted); margin-bottom: 3px; }
.rs-source-text { font-size: 13px; line-height: 1.5; white-space: pre-wrap; max-height: 10.5em; overflow-y: auto; }

.rs-footnote { font-family: var(--rs-mono); font-size: 11px; color: var(--rs-muted); }

@media (max-width: 760px) {
  .rs-stepper { grid-template-columns: 1fr; }
  .rs-step-meta { white-space: normal; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="rs-eyebrow">{escape(eyebrow)}</div>'
        f'<h1 class="rs-title">{escape(title)}</h1>'
        f'<div class="rs-subtitle">{escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def _state(i: int, current: int) -> tuple[str, str]:
    if i < current:
        return "is-done", "Complete"
    if i == current:
        return "is-active", "In progress"
    return "is-locked", "Locked"


def stepper(stages: list[str], current: int, meta: dict[int, str]) -> None:
    items = []
    for i, name in enumerate(stages):
        cls, label = _state(i, current)
        detail = escape(meta.get(i, "")) or "&nbsp;"
        items.append(
            f'<div class="rs-step {cls}"><div class="rs-step-head">'
            f'<span class="rs-step-index">{i + 1:02d}</span><span class="rs-step-name">{escape(name)}</span></div>'
            f'<div class="rs-step-state">{label}</div><div class="rs-step-meta">{detail}</div></div>'
        )
    st.markdown(f'<div class="rs-stepper">{"".join(items)}</div>', unsafe_allow_html=True)


def panel_head(index: int, title: str, current: int, description: str = "", summary: str = "") -> None:
    cls, label = _state(index, current)
    html = (
        f'<div class="rs-panel-head"><span class="rs-panel-index">{index + 1:02d}</span>'
        f'<span class="rs-panel-title">{escape(title)}</span>'
        f'<span class="rs-pill {cls}">{label}</span></div>'
    )
    if summary:
        html += f'<div class="rs-panel-summary">{escape(summary)}</div>'
    elif description:
        html += f'<div class="rs-panel-desc">{escape(description)}</div>'
    if cls == "is-locked":
        html = f'<div class="rs-locked">{html}</div>'
    st.markdown(html, unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="rs-section">{escape(label)}</div>', unsafe_allow_html=True)


def stats(items: list[tuple[str, str] | tuple[str, str, str]]) -> None:
    """Row of stat tiles: (label, value) or (label, value, unit)."""
    tiles = []
    for item in items:
        label, value, unit = (*item, "")[:3]
        unit_html = f'<span class="rs-stat-unit">{escape(unit)}</span>' if unit else ""
        tiles.append(
            f'<div class="rs-stat"><div class="rs-stat-label">{escape(label)}</div>'
            f'<div class="rs-stat-value">{escape(str(value))}{unit_html}</div></div>'
        )
    st.markdown(f'<div class="rs-stats">{"".join(tiles)}</div>', unsafe_allow_html=True)


def spec_sheet(rows: list[tuple[str, str | None, str]]) -> None:
    """Rows of (label, value, detail); a missing value renders as "Not set"."""
    body = "".join(
        f"<dt>{escape(k)}</dt>"
        + (f'<dd>{escape(v)}{f"<small>{escape(d)}</small>" if d else ""}</dd>' if v
           else '<dd class="is-empty">Not set</dd>')
        for k, v, d in rows
    )
    st.markdown(f'<dl class="rs-spec">{body}</dl>', unsafe_allow_html=True)


def sources(hits: list[tuple[int, float]], texts: list[str], pages: list[int]) -> None:
    blocks = "".join(
        f'<div class="rs-source"><div class="rs-source-meta">[Chunk {n}] &middot; page {pages[i]} '
        f'&middot; chunk #{i} &middot; score {s:.3f}</div>'
        f'<div class="rs-source-text">{escape(texts[i])}</div></div>'
        for n, (i, s) in enumerate(hits)
    )
    st.markdown(blocks, unsafe_allow_html=True)


def footnote(text: str) -> None:
    st.markdown(f'<div class="rs-footnote">{escape(text)}</div>', unsafe_allow_html=True)
