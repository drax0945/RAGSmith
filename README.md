# RAGSmith

A minimal Retrieval-Augmented Generation (RAG) pipeline built from scratch, without a RAG
framework (no LangChain/LlamaIndex) -- the goal was to understand what each moving piece
actually does.

## The interactive pipeline (UI)

RAGSmith's UI turns the pipeline into a human-in-the-loop workflow. Each stage is
gated: pick a technique, tune its parameters, run it, inspect the output, and only
then accept it to unlock the next stage. Changing an earlier stage discards
everything after it.

| Stage | Techniques you can choose |
|---|---|
| 1. Load PDF | page-by-page text extraction (pypdf), with page preview |
| 2. Chunk | fixed-size window, sentence packing, paragraph, recursive |
| 3. Embed | Ollama (local), Voyage AI (remote), TF-IDF (local, lexical) |
| 4. Retrieve | cosine similarity, MMR (diverse), hybrid dense + BM25, with test queries |
| 5. Chat | any local Ollama LLM, with page-cited sources for every answer |

```bash
.venv/bin/streamlit run app.py
```

For fully local embeddings, pull an embedding model once: `ollama pull nomic-embed-text`.
Voyage needs `VOYAGE_API_KEY` in `.env`. TF-IDF needs nothing.

New techniques are entries in the `CHUNKERS` / `EMBEDDERS` / `RETRIEVERS` registries
(see `options.py`). The UI builds its pickers and parameter controls from them.

## How it works

```
documents -> chunk -> embed -> store (vectors)
                                     |
query -> embed -> similarity search -> top-k chunks -> prompt -> LLM -> answer
```

1. **Chunking** (`chunking.py`) -- splits a document into overlapping, fixed-size text
   chunks. Overlap prevents an idea from being cut cleanly at a chunk boundary and losing
   context on both sides.
2. **Embeddings** (`embeddings.py`) -- turns text into vectors using
   [Voyage AI](https://voyageai.com)'s `voyage-3` model, so semantic similarity can be
   measured as vector distance. Documents and search queries are embedded separately
   (`input_type="document"` vs `"query"`), since Voyage's models are trained
   asymmetrically for retrieval.
3. **Vector store** (`vectorstore.py`) -- a small in-memory store (plain numpy, no
   external vector DB) that ranks stored chunks against a query vector by cosine
   similarity.
4. **Generation** (`generation.py`) -- assembles the retrieved chunks and the question
   into a grounded prompt ("answer using only this context") and calls an LLM to produce
   the final answer. Runs locally via [Ollama](https://ollama.com) (`llama3.2`) by
   default, so no API key is required to try it end-to-end.
5. **`main.py`** -- wires the above into a CLI loop: index a document once, then ask it
   questions repeatedly.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add a `VOYAGE_API_KEY` (free tier at
[dash.voyageai.com](https://dash.voyageai.com)) -- this is required, since embeddings
are always remote. `ANTHROPIC_API_KEY` is optional; it's not used by default (generation
runs locally via Ollama) but is there if you want to swap `generation.py` to call Claude
instead.

Install and start Ollama, then pull a model:

```bash
brew install ollama
brew services start ollama
ollama pull llama3.2
```

## Run

```bash
.venv/bin/python main.py
```

This indexes `data/sample.txt` and drops you into a prompt where you can ask questions
about it. Empty line to quit.

To index your own document, change `DOC_PATH` in `main.py`.

## Project structure

```
chunking.py       sliding-window text splitting with overlap
embeddings.py      Voyage AI embeddings (asymmetric query/document)
vectorstore.py     in-memory numpy store + cosine similarity search
generation.py      grounded prompt assembly + local LLM call (Ollama)
main.py            ties it all together into an index-then-query CLI loop
app.py             Streamlit UI: the step-by-step, human-in-the-loop pipeline
ui.py              UI styling and HTML components (stepper, stage panels, stat tiles)
charts.py          Altair charts for inspecting each stage's output
loaders.py         PDF text extraction with page tracking
options.py         Technique/Param definitions shared by the stage registries
data/sample.txt    sample document used for testing
```
