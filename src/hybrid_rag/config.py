from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    generation_model: str = os.getenv("GENERATION_MODEL", "qwen3:8b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    index_dir: Path = Path(os.getenv("INDEX_DIR", ".rag_index"))
    vector_store_dir: Path = Path(os.getenv("VECTOR_STORE_DIR", ".chroma"))
    vector_store: str = os.getenv("VECTOR_STORE", "chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "hybrid_rag_chunks")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    bm25_k: int = int(os.getenv("BM25_K", "30"))
    dense_k: int = int(os.getenv("DENSE_K", "30"))
    fused_k: int = int(os.getenv("FUSED_K", "15"))
    rerank_k: int = int(os.getenv("RERANK_K", "6"))
    rrf_constant: int = int(os.getenv("RRF_CONSTANT", "60"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.4"))
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
