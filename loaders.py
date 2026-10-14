"""
Step 0: Loading.

Extract text from a PDF page by page, remembering where each page starts in
the combined text so chunks can later be traced back to page numbers.
"""

import bisect
import io
from dataclasses import dataclass
from typing import Callable

from pypdf import PdfReader

PAGE_SEPARATOR = "\n\n"


@dataclass
class Document:
    name: str
    text: str
    page_starts: list[int]  # character offset where each page begins in `text`

    @property
    def num_pages(self) -> int:
        return len(self.page_starts)

    def page_of(self, offset: int) -> int:
        """1-based page number containing character `offset`."""
        return max(bisect.bisect_right(self.page_starts, offset), 1)


def load_pdf(
    data: bytes,
    name: str = "document.pdf",
    on_progress: Callable[[int, int], None] | None = None,
) -> Document:
    reader = PdfReader(io.BytesIO(data))
    total = len(reader.pages)

    parts: list[str] = []
    page_starts: list[int] = []
    offset = 0
    for i, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()
        page_starts.append(offset)
        parts.append(page_text)
        offset += len(page_text) + len(PAGE_SEPARATOR)
        if on_progress:
            on_progress(i + 1, total)

    return Document(name=name, text=PAGE_SEPARATOR.join(parts), page_starts=page_starts)


def load_text_file(path: str) -> Document:
    with open(path) as f:
        text = f.read()
    return Document(name=path, text=text, page_starts=[0])
