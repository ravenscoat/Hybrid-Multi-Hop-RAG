from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .models import Chunk


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_words(text: str, size: int = 350, overlap: int = 60) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = re.findall(r"\S+", text)
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step)]


def load_chunks(root: Path) -> list[Chunk]:
    files: Iterable[Path] = (
        path for path in sorted(root.rglob("*")) if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    chunks: list[Chunk] = []
    for path in files:
        relative = str(path.relative_to(root))
        for ordinal, text in enumerate(chunk_words(read_document(path))):
            if text.strip():
                chunks.append(Chunk(len(chunks), relative, text, ordinal))
    return chunks

