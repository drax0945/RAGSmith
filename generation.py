"""
Step 4: Generation.

Take the user's question plus the chunks retrieved for it, assemble a prompt
that grounds the model in that context, and call an LLM (running locally via
Ollama) to produce the final answer.
"""

from typing import Iterator

import ollama

MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a question-answering assistant. Answer the user's \
question using ONLY the context provided below. If the answer is not \
contained in the context, say "I don't know based on the provided context" \
-- do not use outside knowledge or make anything up. When you use a chunk, \
cite it like [Chunk 2]."""


def build_prompt(query: str, chunks: list[str]) -> str:
    """
    Assemble the retrieved chunks and the question into a single prompt.

    This is the "augmented" part of Retrieval-Augmented Generation: instead
    of asking the model cold, we hand it the specific evidence it needs so
    it can answer accurately instead of guessing from parametric memory.
    """
    context = "\n\n".join(f"[Chunk {i}]\n{c}" for i, c in enumerate(chunks))
    return f"Context:\n{context}\n\nQuestion: {query}"


def _messages(query, chunks, system_prompt, history):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": build_prompt(query, chunks)})
    return messages


def generate_answer(
    query: str,
    chunks: list[str],
    model: str = MODEL,
    temperature: float = 0.2,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Call the local LLM with the grounded prompt and return its answer."""
    response = ollama.chat(
        model=model,
        messages=_messages(query, chunks, system_prompt, None),
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def stream_answer(
    query: str,
    chunks: list[str],
    model: str = MODEL,
    temperature: float = 0.2,
    system_prompt: str = SYSTEM_PROMPT,
    history: list[dict] | None = None,
) -> Iterator[str]:
    """Same as generate_answer, but yields the answer token by token."""
    stream = ollama.chat(
        model=model,
        messages=_messages(query, chunks, system_prompt, history),
        options={"temperature": temperature},
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]


if __name__ == "__main__":
    chunks = [
        "On the morning of December 17, 1903, Orville piloted the first "
        "flight, covering 120 feet in 12 seconds. Wilbur made the fourth "
        "and longest flight that day, covering 852 feet in 59 seconds.",
    ]
    query = "How long was the first flight?"

    answer = generate_answer(query, chunks)
    print(f"Q: {query}\n")
    print(f"A: {answer}")
