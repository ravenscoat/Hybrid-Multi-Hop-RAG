#!/usr/bin/env python3
"""Audit a code graph, remove duplicate edges, and rank a query anchor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_rag.code_agent import CodeKnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--knowledge-base", type=Path, default=Path("outputs/code_knowledge_base.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/code_knowledge_base_clean.json"))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    kb = CodeKnowledgeBase(args.knowledge_base)
    audit = kb.audit_and_deduplicate()
    ranking = kb.anchor_and_pagerank(args.query, args.limit)
    payload = json.loads(args.knowledge_base.read_text(encoding="utf-8"))
    payload["nodes"] = list(kb.nodes.values())
    payload["edges"] = kb.edges
    payload["stats"] = {"nodes": len(kb.nodes), "edges": len(kb.edges)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "query": args.query, **ranking, "clean_graph": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
