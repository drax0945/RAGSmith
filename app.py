"""
RAGSmith UI: a human-in-the-loop RAG pipeline.

Each stage (ingest -> chunk -> embed -> retrieve -> generate) is gated: you
pick a technique and its parameters, run it, inspect the output, and only
then accept it to unlock the next stage. Revising an earlier stage discards
everything downstream of it.

    streamlit run app.py
"""

import json
import time

import numpy as np
import pandas as pd
import streamlit as st

import charts
import ui
from chunking import CHUNKERS, locate_chunks
from embeddings import (
    EMBEDDERS,
    PULLABLE_EMBEDDERS,
    ollama_embedding_models,
    ollama_models,
    pull_ollama_model,
)
from generation import SYSTEM_PROMPT, stream_answer
from loaders import load_pdf
from options import Technique
from vectorstore import RETRIEVERS, VectorStore

STAGES = ["Ingest", "Chunk", "Embed", "Retrieve", "Generate"]
LOAD, CHUNK, EMBED, RETRIEVE, CHAT = range(len(STAGES))
DESCRIPTIONS = {
    LOAD: "Extract text from the source PDF page by page and check extraction quality.",
    CHUNK: "Split the text into retrievable units. Size and boundaries decide what the model can see at once.",
    EMBED: "Encode every chunk as a vector so relevance can be measured as similarity.",
    RETRIEVE: "Choose how chunks are ranked for a query, and test it before committing.",
    CHAT: "Ask questions. Answers come from the local model, grounded in the retrieved chunks.",
}
LOW_TEXT_PAGE = 200  # characters; fewer than this usually means a scanned page
AVATARS = {"user": ":material/person:", "assistant": ":material/subject:"}

st.set_page_config(page_title="RAGSmith", layout="wide")
ui.inject_css()

ss = st.session_state
ss.setdefault("stage", LOAD)
ss.setdefault("results", {})  # stage index -> output of that stage
ss.setdefault("choices", {})  # stage index -> short summary of the accepted choice
ss.setdefault("config", {})  # stage index -> structured config, for export
ss.setdefault("messages", [])
ss.setdefault("query_cache", {})


# ---------------------------------------------------------------- state


def reopen(stage: int) -> None:
    """Go back to `stage`, discarding everything downstream of it."""
    ss.stage = stage
    for later in range(stage, len(STAGES)):
        ss.choices.pop(later, None)
        ss.config.pop(later, None)
        if later > stage:
            ss.results.pop(later, None)
    ss.messages = []
    ss.query_cache = {}


def accept(stage: int, summary: str, config: dict) -> None:
    ss.choices[stage] = summary
    ss.config[stage] = config
    ss.stage = stage + 1


# ---------------------------------------------------------------- widgets


def render_params(technique: Technique, key: str) -> dict:
    values = {}
    for p in technique.params:
        wkey = f"{key}.{technique.name}.{p.name}"
        if p.choices is not None:
            idx = p.choices.index(p.default) if p.default in p.choices else 0
            values[p.name] = st.selectbox(p.label, p.choices, index=idx, key=wkey, help=p.help or None)
        elif isinstance(p.default, float):
            values[p.name] = st.slider(p.label, float(p.min), float(p.max), float(p.default), float(p.step), key=wkey)
        else:
            values[p.name] = st.slider(p.label, int(p.min), int(p.max), int(p.default), int(p.step), key=wkey)
    return values


def technique_picker(registry: dict[str, Technique], key: str) -> Technique:
    names = list(registry)
    name = st.radio(
        "Technique", names, key=f"{key}.technique",
        captions=[registry[n].description for n in names], label_visibility="collapsed",
    )
    return registry[name]


def fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def stage_panel(stage: int):
    """
    Render a stage's panel. Completed and locked stages collapse to a single
    header row; the active stage returns its container for the caller to fill.
    """
    box = st.container(border=True)
    with box:
        if stage < ss.stage:
            left, right = st.columns([6, 1], vertical_alignment="center")
            with left:
                ui.panel_head(stage, STAGES[stage], ss.stage, summary=ss.choices.get(stage, ""))
            if right.button("Revise", key=f"reopen.{stage}", width="stretch",
                            help="Reopen this stage. Later stages are reset."):
                reopen(stage)
                st.rerun()
            return None
        if stage > ss.stage:
            ui.panel_head(stage, STAGES[stage], ss.stage, description=DESCRIPTIONS[stage])
            return None
        ui.panel_head(stage, STAGES[stage], ss.stage, description=DESCRIPTIONS[stage])
    return box


def continue_row(stage: int, enabled: bool, summary: str, config: dict) -> None:
    st.divider()
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        ui.footnote(
            f"Accepting locks this configuration and unlocks {STAGES[stage + 1]}."
            if enabled else "Run this stage to review its output before continuing."
        )
    if right.button(f"Continue to {STAGES[stage + 1]}", type="primary", disabled=not enabled,
                    key=f"accept.{stage}", width="stretch"):
        accept(stage, summary, config)
        st.rerun()


def embed_query_cached(embedder, query: str) -> np.ndarray:
    key = (id(embedder), query)
    if key not in ss.query_cache:
        ss.query_cache[key] = embedder.embed_query(query)
    return ss.query_cache[key]


@st.cache_data(show_spinner=False)
def compare_chunkers(text: str) -> pd.DataFrame:
    rows = []
    for name, t in CHUNKERS.items():
        chunks = [c for c in t.build(text, **t.defaults()) if c.strip()]
        lengths = np.array([len(c) for c in chunks]) if chunks else np.array([0])
        rows.append({
            "Technique": name,
            "Defaults": fmt_params(t.defaults()),
            "Chunks": len(chunks),
            "Mean size": int(lengths.mean()),
            "Min": int(lengths.min()),
            "Max": int(lengths.max()),
            "Coverage": round(lengths.sum() / max(len(text), 1), 2),
        })
    return pd.DataFrame(rows)


def empty_state(text: str) -> None:
    st.markdown(f'<div class="rs-panel-desc" style="padding:28px 0">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown('<div class="rs-brand">RAGSmith</div>'
                '<div class="rs-brand-sub">Retrieval pipeline workbench</div>', unsafe_allow_html=True)
    ui.section("Configuration")

    def spec(stage: int) -> tuple[str | None, str]:
        cfg = ss.config.get(stage)
        if not cfg:
            return None, ""
        if stage == LOAD:
            return cfg["file"], f"{cfg['pages']} pages"
        return cfg["technique"], fmt_params(cfg["params"])

    ui.spec_sheet([
        ("Document", *spec(LOAD)),
        ("Chunking", *spec(CHUNK)),
        ("Embedding", *spec(EMBED)),
        ("Retrieval", *spec(RETRIEVE)),
        ("Model", ss.get("chat.model") if ss.stage == CHAT else None, ""),
    ])
    st.write("")
    ui.section("Session")
    st.download_button(
        "Export configuration",
        json.dumps({STAGES[k]: v for k, v in sorted(ss.config.items())}, indent=2, default=str),
        file_name="ragsmith-config.json", mime="application/json",
        width="stretch", disabled=not ss.config,
    )
    if st.button("Reset pipeline", width="stretch"):
        ss.clear()
        st.rerun()

# ---------------------------------------------------------------- header

doc = ss.results.get(LOAD)
ui.page_header(
    "RAGSmith / Pipeline",
    doc.name if doc and ss.stage > LOAD else "New pipeline",
    "Configure each stage, inspect what it produced, and accept it to unlock the next.",
)
ui.stepper(STAGES, ss.stage, ss.choices)

# ---------------------------------------------------------------- 1. ingest

if (panel := stage_panel(LOAD)) is not None:
    with panel:
        left, right = st.columns([1, 2], gap="large")
        with left:
            ui.section("Source")
            upload = st.file_uploader("PDF document", type=["pdf"])
            if st.button("Extract text", type="primary", disabled=upload is None, width="stretch"):
                bar = st.progress(0.0, text="Reading pages")
                doc = load_pdf(
                    upload.getvalue(), upload.name,
                    on_progress=lambda d, n: bar.progress(d / n, text=f"Page {d} of {n}"),
                )
                bar.empty()
                ss.results[LOAD] = doc

        with right:
            ui.section("Extraction")
            doc = ss.results.get(LOAD)
            if not doc:
                empty_state("Upload a PDF and extract its text to see page-level statistics.")
            else:
                ends = doc.page_starts[1:] + [len(doc.text) + 2]
                per_page = [e - s - 2 for s, e in zip(doc.page_starts, ends)]
                low = sum(1 for n in per_page if n < LOW_TEXT_PAGE)
                ui.stats([
                    ("Pages", f"{doc.num_pages:,}"),
                    ("Characters", f"{len(doc.text):,}"),
                    ("Words", f"{len(doc.text.split()):,}"),
                    ("Per page", f"{np.mean(per_page):,.0f}", "chars"),
                    ("Low-text pages", f"{low:,}"),
                ])
                density, viewer = st.tabs(["Text density", "Page viewer"])
                with density:
                    st.altair_chart(charts.page_density(per_page), width="stretch")
                    ui.footnote(f"Pages under {LOW_TEXT_PAGE} characters are usually scanned images "
                                "and need OCR before they can be retrieved.")
                with viewer:
                    page = st.number_input("Page", 1, doc.num_pages, 1)
                    s, e = doc.page_starts[page - 1], ends[page - 1] - 2
                    st.text_area("Page text", doc.text[s:e], height=260, disabled=True,
                                 label_visibility="collapsed")

        ok = bool(doc and doc.text.strip())
        if doc and not ok:
            st.error("No text could be extracted. The PDF is likely image-only and needs OCR.")
        continue_row(LOAD, ok, f"{doc.name}, {doc.num_pages} pages" if doc else "",
                     {"file": doc.name if doc else None, "pages": doc.num_pages if doc else None})

# ---------------------------------------------------------------- 2. chunk

if (panel := stage_panel(CHUNK)) is not None:
    doc = ss.results[LOAD]
    with panel:
        left, right = st.columns([1, 2], gap="large")
        with left:
            ui.section("Technique")
            technique = technique_picker(CHUNKERS, "chunk")
            ui.section("Parameters")
            params = render_params(technique, "chunk")
            if st.button("Run chunking", type="primary", width="stretch"):
                try:
                    chunks = technique.build(doc.text, **params)
                except ValueError as e:
                    st.error(str(e))
                else:
                    chunks = [c for c in chunks if c.strip()]
                    ss.results[CHUNK] = {
                        "chunks": chunks,
                        "pages": [doc.page_of(o) for o in locate_chunks(doc.text, chunks)],
                        "technique": technique.name,
                        "params": params,
                    }

        with right:
            ui.section("Output")
            out = ss.results.get(CHUNK)
            if not out:
                empty_state("Choose a technique and run it to inspect the resulting chunks.")
            else:
                chunks, pages = out["chunks"], out["pages"]
                lengths = np.array([len(c) for c in chunks])
                ui.stats([
                    ("Chunks", f"{len(chunks):,}"),
                    ("Mean", f"{lengths.mean():,.0f}", "chars"),
                    ("Median", f"{np.median(lengths):,.0f}", "chars"),
                    ("Range", f"{lengths.min():,}–{lengths.max():,}"),
                    ("Coverage", f"{lengths.sum() / len(doc.text):.2f}", "x"),
                ])
                dist, inspect, table, compare = st.tabs(
                    ["Size distribution", "Inspector", "All chunks", "Compare techniques"])
                with dist:
                    st.altair_chart(charts.size_histogram(lengths.tolist()), width="stretch")
                    ui.footnote("Coverage above 1.0 is text repeated through overlap.")
                with inspect:
                    i = st.number_input("Chunk", 0, len(chunks) - 1, 0)
                    ui.footnote(f"Chunk {i} of {len(chunks) - 1}  ·  page {pages[i]}  ·  "
                                f"{len(chunks[i]):,} characters  ·  {len(chunks[i].split()):,} words")
                    st.text_area("Chunk text", chunks[i], height=220, disabled=True,
                                 label_visibility="collapsed")
                with table:
                    st.dataframe(
                        pd.DataFrame({
                            "Chunk": range(len(chunks)), "Page": pages, "Characters": lengths,
                            "Preview": [c[:160].replace("\n", " ") for c in chunks],
                        }),
                        hide_index=True, height=320,
                        column_config={"Preview": st.column_config.TextColumn(width="large")},
                    )
                with compare:
                    st.dataframe(compare_chunkers(doc.text), hide_index=True)
                    ui.footnote("Every technique run with its default parameters on this document.")

        if out:
            summary = f"{out['technique']} ({fmt_params(out['params'])}), {len(out['chunks']):,} chunks"
        continue_row(CHUNK, bool(out), summary if out else "",
                     {"technique": out["technique"], "params": out["params"]} if out else {})

# ---------------------------------------------------------------- 3. embed

if (panel := stage_panel(EMBED)) is not None:
    chunks, pages = ss.results[CHUNK]["chunks"], ss.results[CHUNK]["pages"]
    with panel:
        left, right = st.columns([1, 2], gap="large")
        with left:
            ui.section("Technique")
            EMBEDDERS["Ollama (local)"].params[0].choices = ollama_embedding_models()
            technique = technique_picker(EMBEDDERS, "embed")

            blocked = technique.name.startswith("Ollama") and not technique.params[0].choices
            ui.section("Parameters")
            if blocked:
                st.warning("No local embedding model is installed. Download one, "
                           "or choose TF-IDF to continue without a model.")
                to_pull = st.selectbox("Model to download", list(PULLABLE_EMBEDDERS),
                                       format_func=lambda m: f"{m}  ({PULLABLE_EMBEDDERS[m]})")
                if st.button(f"Download {to_pull}", width="stretch"):
                    bar = st.progress(0.0, text=f"Downloading {to_pull}")
                    try:
                        for update in pull_ollama_model(to_pull):
                            # Ollama sends total before completed, and status-only updates
                            # (e.g. "verifying sha256 digest") with neither.
                            done, total = update.completed or 0, update.total or 0
                            if total:
                                bar.progress(min(done / total, 1.0),
                                             text=f"{update.status}: {done / 1e6:.0f} of {total / 1e6:.0f} MB")
                            else:
                                bar.progress(1.0 if update.status == "success" else 0.0, text=update.status)
                    except Exception as e:
                        bar.empty()
                        st.error(f"Download failed: {e}")
                    else:
                        st.rerun()
                params = {}
            else:
                params = render_params(technique, "embed")

            if st.button(f"Embed {len(chunks):,} chunks", type="primary", disabled=blocked, width="stretch"):
                bar = st.progress(0.0, text="Embedding")
                t0 = time.time()
                try:
                    embedder = technique.build(**params)
                    vectors = embedder.embed_documents(
                        chunks, on_progress=lambda d, n: bar.progress(d / n, text=f"Embedded {d:,} of {n:,}"))
                except Exception as e:
                    bar.empty()
                    st.error(f"Embedding failed: {e}")
                else:
                    elapsed = time.time() - t0
                    bar.progress(1.0, text="Projecting vectors")
                    store = VectorStore()
                    store.add(chunks, vectors)
                    proj, variance = None, None
                    if len(chunks) >= 3:
                        from sklearn.decomposition import PCA

                        sample = np.arange(len(chunks))
                        if len(sample) > 2000:
                            sample = np.sort(np.random.default_rng(0).choice(sample, 2000, replace=False))
                        pca = PCA(n_components=2, random_state=0)
                        normed = vectors[sample] / np.maximum(np.linalg.norm(vectors[sample], axis=1, keepdims=True), 1e-12)
                        xy = pca.fit_transform(normed)
                        proj = pd.DataFrame({
                            "x": xy[:, 0], "y": xy[:, 1], "chunk": sample,
                            "page": [pages[i] for i in sample],
                            "preview": [chunks[i][:120].replace("\n", " ") for i in sample],
                        })
                        variance = float(pca.explained_variance_ratio_.sum())
                    bar.empty()
                    ss.results[EMBED] = {
                        "embedder": embedder, "store": store, "seconds": elapsed,
                        "technique": technique.name, "params": params,
                        "projection": proj, "variance": variance,
                    }

        with right:
            ui.section("Output")
            out = ss.results.get(EMBED)
            if not out:
                empty_state("Choose an embedding technique and run it to inspect the vector space.")
            else:
                vecs = out["store"].vectors
                ui.stats([
                    ("Vectors", f"{vecs.shape[0]:,}"),
                    ("Dimensions", f"{vecs.shape[1]:,}"),
                    ("Elapsed", f"{out['seconds']:.1f}", "s"),
                    ("Throughput", f"{vecs.shape[0] / max(out['seconds'], 1e-3):,.0f}", "chunks/s"),
                    ("Index size", *((f"{vecs.nbytes / 1e6:,.1f}", "MB") if vecs.nbytes >= 1e5
                                     else (f"{vecs.nbytes / 1e3:,.1f}", "KB"))),
                ])
                proj_tab, nn_tab = st.tabs(["Projection", "Nearest neighbours"])
                with proj_tab:
                    if out["projection"] is None:
                        empty_state("At least three chunks are needed for a projection.")
                    else:
                        st.altair_chart(charts.projection(out["projection"]), width="stretch")
                        ui.footnote(f"PCA projection of normalised vectors; {out['variance']:.0%} of variance shown. "
                                    "Nearby points are similar chunks. Scroll to zoom.")
                with nn_tab:
                    i = st.number_input("Reference chunk", 0, len(chunks) - 1, 0, key="nn.chunk")
                    ui.footnote(f"Page {pages[i]}  ·  {chunks[i][:220]}")
                    sims = out["store"].cosine_scores(vecs[i])
                    order = [j for j in np.argsort(sims)[::-1] if j != i][:8]
                    st.dataframe(
                        pd.DataFrame({
                            "Chunk": order, "Page": [pages[j] for j in order],
                            "Similarity": [float(sims[j]) for j in order],
                            "Preview": [chunks[j][:160].replace("\n", " ") for j in order],
                        }),
                        hide_index=True,
                        column_config={
                            "Similarity": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0),
                            "Preview": st.column_config.TextColumn(width="large"),
                        },
                    )

        if out:
            summary = out["embedder"].name
        continue_row(EMBED, bool(out), summary if out else "",
                     {"technique": out["technique"], "params": out["params"]} if out else {})

# ---------------------------------------------------------------- 4. retrieve

if (panel := stage_panel(RETRIEVE)) is not None:
    store, embedder = ss.results[EMBED]["store"], ss.results[EMBED]["embedder"]
    pages = ss.results[CHUNK]["pages"]
    with panel:
        left, right = st.columns([1, 2], gap="large")
        with left:
            ui.section("Technique")
            technique = technique_picker(RETRIEVERS, "retrieve")
            ui.section("Parameters")
            params = render_params(technique, "retrieve")

        with right:
            ui.section("Test query")
            query = st.text_input("Query", placeholder="Ask something the document should answer",
                                  label_visibility="collapsed")
            if not query:
                empty_state("Enter a query to see what this configuration retrieves.")
            else:
                t0 = time.time()
                qvec = embed_query_cached(embedder, query)
                hits = technique.build(store, qvec, query, **params)
                latency = (time.time() - t0) * 1000
                ui.stats([
                    ("Retrieved", f"{len(hits)}", "chunks"),
                    ("Top score", f"{hits[0][1]:.3f}" if hits else "–"),
                    ("Pages", f"{len({pages[i] for i, _ in hits})}", "distinct"),
                    ("Latency", f"{latency:,.0f}", "ms"),
                ])
                results_tab, profile_tab, compare_tab = st.tabs(["Results", "Score profile", "Compare techniques"])
                with results_tab:
                    ui.sources(hits, store.texts, pages)
                with profile_tab:
                    st.altair_chart(charts.score_profile(pd.DataFrame({
                        "rank": range(1, len(hits) + 1), "chunk": [i for i, _ in hits],
                        "page": [pages[i] for i, _ in hits], "score": [s for _, s in hits],
                    })), width="stretch")
                    ui.footnote("Scores are comparable within a technique, not across techniques.")
                with compare_tab:
                    k = min(params["k"], len(store.texts))
                    ranked = {
                        name: [i for i, _ in t.build(store, qvec, query, **{**t.defaults(), "k": k})]
                        for name, t in RETRIEVERS.items()
                    }
                    st.dataframe(
                        pd.DataFrame({"Rank": range(1, k + 1), **{n: pd.Series(v) for n, v in ranked.items()}}),
                        hide_index=True,
                    )
                    base = set(ranked[technique.name])
                    ui.footnote("Chunk ids per rank, default parameters, same k.  Overlap with selected: " + ",  ".join(
                        f"{n} {len(base & set(v))}/{k}" for n, v in ranked.items() if n != technique.name))

        ss.results[RETRIEVE] = {"technique": technique, "params": params}
        continue_row(RETRIEVE, True, f"{technique.name} ({fmt_params(params)})",
                     {"technique": technique.name, "params": params})

# ---------------------------------------------------------------- 5. generate

if (panel := stage_panel(CHAT)) is not None:
    store, embedder = ss.results[EMBED]["store"], ss.results[EMBED]["embedder"]
    pages = ss.results[CHUNK]["pages"]
    retriever = ss.results[RETRIEVE]

    embed_only = set(ollama_embedding_models())
    llms = [m for m in ollama_models() if m not in embed_only]
    with panel:
        if not llms:
            st.error("No local Ollama chat model is available. Start Ollama and pull a model:")
            st.code("ollama pull llama3.2", language="bash")
            st.stop()

        left, right = st.columns([1, 2], gap="large")
        with left:
            ui.section("Model")
            model = st.selectbox("Local LLM", llms, key="chat.model")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
            remember = st.toggle("Include conversation history", value=False,
                                 help="Send previous turns to the model so follow-up questions work.")
            with st.expander("System prompt"):
                system_prompt = st.text_area("System prompt", SYSTEM_PROMPT, height=160,
                                             label_visibility="collapsed")
            ui.section("Retrieval")
            ui.footnote(f"{retriever['technique'].name}  ·  {fmt_params(retriever['params'])}")
            st.write("")
            if st.button("Clear conversation", width="stretch", disabled=not ss.messages):
                ss.messages = []
                st.rerun()

        with right:
            ui.section("Conversation")
            history_box = st.container(height=560, border=False)

            def render_message(msg: dict) -> None:
                with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
                    st.markdown(msg["content"])
                    if msg.get("hits"):
                        with st.expander(f"Sources  ·  {len(msg['hits'])} chunks"):
                            ui.sources(msg["hits"], store.texts, pages)
                    if msg.get("timing"):
                        ui.footnote(msg["timing"])

            with history_box:
                if not ss.messages:
                    empty_state("Ask a question about the document. Each answer lists the chunks it was grounded in.")
                for msg in ss.messages:
                    render_message(msg)

            question = st.chat_input("Ask about the document")

        if question:
            t0 = time.time()
            hits = retriever["technique"].build(store, embedder.embed_query(question), question,
                                                **retriever["params"])
            t_retrieve = (time.time() - t0) * 1000
            history = [{"role": m["role"], "content": m["content"]} for m in ss.messages] if remember else None
            with history_box:
                render_message({"role": "user", "content": question})
                with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                    t1 = time.time()
                    try:
                        answer = st.write_stream(stream_answer(
                            question, [store.texts[i] for i, _ in hits], model, temperature,
                            system_prompt, history))
                    except Exception as e:
                        answer = f"Generation failed: {e}"
                        st.error(answer)
            timing = f"{model}  ·  retrieval {t_retrieve:,.0f} ms  ·  generation {time.time() - t1:.1f} s"
            ss.messages.append({"role": "user", "content": question})
            ss.messages.append({"role": "assistant", "content": answer, "hits": hits, "timing": timing})
            st.rerun()
