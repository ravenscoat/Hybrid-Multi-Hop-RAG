from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from usearch.index import Index

from .bm25 import BM25
from .models import Chunk


class HybridIndex:
    def __init__(self, directory: Path, dimension: int):
        self.directory = directory
        self.dimension = dimension
        self.chunks: list[Chunk] = []
        self.bm25: BM25 | None = None
        self.hnsw: Index | None = None

    def build(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            raise ValueError("No supported document content was found")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.shape != (len(chunks), self.dimension):
            raise ValueError(f"Expected embedding shape {(len(chunks), self.dimension)}, got {matrix.shape}")
        self.directory.mkdir(parents=True, exist_ok=True)
        index = Index(
            ndim=self.dimension,
            metric="cos",
            dtype="f32",
            connectivity=16,
            expansion_add=100,
            expansion_search=64,
        )
        index.add(np.arange(len(chunks), dtype=np.uint64), matrix)
        index.save(self.directory / "dense.usearch")
        (self.directory / "chunks.json").write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False), encoding="utf-8"
        )
        self.chunks = chunks
        self.bm25 = BM25([chunk.text for chunk in chunks])
        self.hnsw = index

    def load(self) -> None:
        values = json.loads((self.directory / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk.from_dict(value) for value in values]
        self.bm25 = BM25([chunk.text for chunk in self.chunks])
        self.hnsw = Index(
            ndim=self.dimension,
            metric="cos",
            dtype="f32",
            connectivity=16,
            expansion_search=64,
        )
        self.hnsw.load(self.directory / "dense.usearch")

    def keyword(self, query: str, limit: int) -> list[int]:
        assert self.bm25 is not None
        return [doc_id for doc_id, _ in self.bm25.search(query, limit)]

    def dense(self, vector: list[float], limit: int) -> list[int]:
        assert self.hnsw is not None
        actual_limit = min(limit, len(self.chunks))
        matches = self.hnsw.search(np.asarray(vector, dtype=np.float32), actual_limit)
        return [int(value) for value in matches.keys]
