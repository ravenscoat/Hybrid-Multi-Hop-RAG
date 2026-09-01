from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


class CodeKnowledgeBase:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.nodes = {item["id"]: item for item in payload.get("nodes", [])}
        self.edges = payload.get("edges", [])

    def search(self, query: str, edge_type: str | None = None, limit: int = 12) -> dict[str, Any]:
        terms = {term for term in re.findall(r"[a-zA-Z_][\w.]*", query.casefold()) if len(term) > 2}
        matched = [node for node in self.nodes.values() if any(term in json.dumps(node).casefold() for term in terms)]
        matched_ids = {node["id"] for node in matched}
        edges = [edge for edge in self.edges if not edge_type or edge["type"] == edge_type]
        relevant = [edge for edge in edges if edge["source"] in matched_ids or edge["target"] in matched_ids]
        return {"matched_nodes": matched[:limit], "edges": relevant[:limit], "edge_type": edge_type}


class AdaptiveCodeAgent:
    """Route code questions to the smallest useful graph view."""

    ROUTES = {
        "imports": ("import_to_import", ("import", "dependency", "depends", "module")),
        "functions": ("function_to_function", ("function", "call", "calls", "invokes", "method")),
        "history": ("file_to_file", ("git", "history", "changed", "co-change", "together", "commit")),
    }

    def __init__(self, kb: CodeKnowledgeBase):
        self.kb = kb

    def ask(self, question: str, limit: int = 12) -> dict[str, Any]:
        lowered = question.casefold()
        scores = {route: sum(word in lowered for word in words) for route, (_, words) in self.ROUTES.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            best = "functions" if any(token in lowered for token in ("impact", "architecture", "flow")) else "imports"
        edge_type = self.ROUTES[best][0]
        result = self.kb.search(question, edge_type=edge_type, limit=limit)
        result.update({"route": best, "route_scores": scores})
        return result
