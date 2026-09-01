#!/usr/bin/env python3
"""Embed code-graph nodes with the configured local Qwen embedding model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_rag.config import Settings
from hybrid_rag.ollama import OllamaClient


def node_text(node: dict) -> str:
    value = node.get("semantic_text")
    if value:
        return str(value)
    return " ".join(str(node.get(key, "")) for key in ("kind", "qualified_name", "name", "file", "path", "module", "signature", "docstring", "source_excerpt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("outputs/code_knowledge_base.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/code_knowledge_base_embeddings.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    settings = Settings()
    client = OllamaClient(settings.ollama_url)
    nodes = graph.get("nodes", [])
    vectors: list[list[float]] = []
    for start in range(0, len(nodes), args.batch_size):
        batch = nodes[start : start + args.batch_size]
        texts = [f"Represent this code symbol for software-maintenance retrieval:\n{node_text(node)}" for node in batch]
        vectors.extend(client.embed(settings.embedding_model, texts))
        print(f"Embedded {min(start + args.batch_size, len(nodes))}/{len(nodes)}", flush=True)
    payload = {
        "model": settings.embedding_model,
        "graph_schema_version": graph.get("schema_version"),
        "embeddings": {node["id"]: vector for node, vector in zip(nodes, vectors)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "nodes": len(vectors), "dimensions": len(vectors[0]) if vectors else 0}, indent=2))


if __name__ == "__main__":
    main()
