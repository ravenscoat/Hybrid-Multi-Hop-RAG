from __future__ import annotations

import json
from pathlib import Path

from .bm25 import BM25
from .models import Chunk


class HybridIndex:
    def __init__(self, directory: Path, dimension: int, collection_name: str = "hybrid_rag_chunks"):
        self.directory = directory
        self.dimension = dimension
        self.collection_name = collection_name
        self.chunks: list[Chunk] = []
        self.bm25: BM25 | None = None
        self.collection = None

    def build(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            raise ValueError("No supported document content was found")
        if len(embeddings) != len(chunks) or any(len(vector) != self.dimension for vector in embeddings):
            raise ValueError(f"Expected {len(chunks)} embeddings of dimension {self.dimension}")
        import chromadb

        self.directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.directory))
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        batch_size = 5000
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[str(chunk.id) for chunk in batch],
                embeddings=embeddings[start : start + batch_size],
                documents=[chunk.text for chunk in batch],
                metadatas=[{"source": chunk.source, "ordinal": chunk.ordinal} for chunk in batch],
            )
        (self.directory / "chunks.json").write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False), encoding="utf-8"
        )
        self.chunks = chunks
        self.bm25 = BM25([chunk.text for chunk in chunks])
        self.collection = collection

    def load(self) -> None:
        values = json.loads((self.directory / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk.from_dict(value) for value in values]
        self.bm25 = BM25([chunk.text for chunk in self.chunks])
        import chromadb

        client = chromadb.PersistentClient(path=str(self.directory))
        self.collection = client.get_collection(self.collection_name)

    def keyword(self, query: str, limit: int) -> list[int]:
        assert self.bm25 is not None
        return [doc_id for doc_id, _ in self.bm25.search(query, limit)]

    def dense(self, vector: list[float], limit: int) -> list[int]:
        if self.collection is None:
            raise RuntimeError("Vector collection is not loaded")
        actual_limit = min(limit, len(self.chunks))
        matches = self.collection.query(query_embeddings=[vector], n_results=actual_limit, include=["metadatas"])
        return [int(value) for value in matches["ids"][0]]
