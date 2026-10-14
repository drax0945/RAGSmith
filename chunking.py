"""
Step 1: Chunking.

Split a document into text chunks that are small enough to embed and retrieve
meaningfully, but large enough to preserve context. Several strategies are
available; each trades off simplicity against respecting natural boundaries.
"""

import re

from options import Param, Technique


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Fixed-size sliding window over raw characters. Each chunk overlaps the
    previous one by `overlap` characters so an idea cut at a boundary still
    appears whole in one of the two neighbours.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step

    return chunks


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_sentences(text: str, max_chars: int = 800, overlap_sentences: int = 1) -> list[str]:
    """
    Pack whole sentences into chunks of up to `max_chars`, never cutting a
    sentence in half. The last `overlap_sentences` of each chunk are repeated
    at the start of the next one.
    """
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        if current and length + len(sentence) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            length = sum(len(s) + 1 for s in current)
        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_paragraphs(text: str, max_chars: int = 1500) -> list[str]:
    """
    Split on blank lines, merging consecutive short paragraphs until
    `max_chars` is reached. Paragraphs longer than `max_chars` are split
    with the fixed-size window.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(chunk_text(para, max_chars, max_chars // 10))
        elif current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current:
        chunks.append(current)
    return chunks


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_recursive(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Recursive splitting: try the coarsest separator (paragraphs) first and
    only fall back to finer ones (lines, sentences, words, characters) for
    pieces that are still too big. Pieces are then greedily merged back up
    to `chunk_size`, carrying `overlap` characters between chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    def split(piece: str, seps: list[str]) -> list[str]:
        if len(piece) <= chunk_size:
            return [piece]
        sep, rest = seps[0], seps[1:]
        if sep == "":
            return [piece[i : i + chunk_size] for i in range(0, len(piece), chunk_size)]
        parts = piece.split(sep)
        out: list[str] = []
        for i, part in enumerate(parts):
            # Keep the separator attached so merged chunks read naturally.
            part = part + sep if i < len(parts) - 1 else part
            if part:
                out.extend(split(part, rest))
        return out

    chunks: list[str] = []
    current = ""
    for piece in split(text, _SEPARATORS):
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


CHUNKERS: dict[str, Technique] = {
    t.name: t
    for t in [
        Technique(
            "Fixed-size window",
            "Equal character windows with overlap. Fast and predictable, but cuts mid-word and mid-sentence.",
            chunk_text,
            [
                Param("chunk_size", "Chunk size (chars)", 800, 100, 5000, 50),
                Param("overlap", "Overlap (chars)", 100, 0, 1000, 10),
            ],
        ),
        Technique(
            "Sentence packing",
            "Packs whole sentences up to a size limit. Never splits a sentence.",
            chunk_sentences,
            [
                Param("max_chars", "Max chunk size (chars)", 800, 100, 5000, 50),
                Param("overlap_sentences", "Overlap (sentences)", 1, 0, 5, 1),
            ],
        ),
        Technique(
            "Paragraph",
            "Uses blank lines as boundaries and merges short paragraphs. Best for well-structured prose.",
            chunk_paragraphs,
            [Param("max_chars", "Max chunk size (chars)", 1500, 200, 6000, 100)],
        ),
        Technique(
            "Recursive",
            "Splits by paragraph → line → sentence → word, only as fine as needed. A good general-purpose default.",
            chunk_recursive,
            [
                Param("chunk_size", "Chunk size (chars)", 800, 100, 5000, 50),
                Param("overlap", "Overlap (chars)", 100, 0, 1000, 10),
            ],
        ),
    ]
}


def locate_chunks(text: str, chunks: list[str]) -> list[int]:
    """
    Best-effort start offset of each chunk in `text` (used to map chunks to
    PDF pages). Chunkers that rejoin pieces may not produce exact substrings,
    so match on a short prefix and fall back to the previous position.
    """
    starts: list[int] = []
    cursor = 0
    for chunk in chunks:
        probe = chunk[:40]
        pos = text.find(probe, cursor)
        if pos == -1:
            pos = text.find(probe)
        if pos == -1:
            pos = cursor
        starts.append(pos)
        cursor = pos + 1
    return starts


if __name__ == "__main__":
    with open("data/sample.txt") as f:
        text = f.read()

    for name, technique in CHUNKERS.items():
        chunks = technique.build(text, **technique.defaults())
        print(f"{name}: {len(chunks)} chunks")
