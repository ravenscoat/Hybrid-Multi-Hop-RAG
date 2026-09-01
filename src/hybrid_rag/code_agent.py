from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class CodeKnowledgeBase:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.nodes = {item["id"]: item for item in payload.get("nodes", [])}
        self.edges = payload.get("edges", [])

    def audit_and_deduplicate(self) -> dict[str, Any]:
        """Remove duplicate nodes/edges and report graph hygiene findings."""
        original_nodes = len(self.nodes)
        original_edges = len(self.edges)
        unique_edges: dict[tuple[str, str, str], dict] = {}
        for edge in self.edges:
            key = (edge.get("type", ""), edge.get("source", ""), edge.get("target", ""))
            current = unique_edges.get(key)
            if current is None or edge.get("cochange_count", 0) > current.get("cochange_count", 0):
                unique_edges[key] = edge
        self.edges = list(unique_edges.values())
        orphan_edges = [edge for edge in self.edges if edge["source"] not in self.nodes or edge["target"] not in self.nodes]
        return {
            "nodes_before": original_nodes,
            "nodes_after": len(self.nodes),
            "edges_before": original_edges,
            "edges_after": len(self.edges),
            "duplicate_edges_removed": original_edges - len(self.edges),
            "orphan_edges": len(orphan_edges),
            "edge_types": dict(Counter(edge.get("type") for edge in self.edges)),
        }

    def anchor_and_pagerank(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Find a lexical query anchor, then rank its graph neighborhood."""
        terms = {term for term in re.findall(r"[a-zA-Z_][\w.]*", query.casefold()) if len(term) > 2}
        scored: list[tuple[float, dict]] = []
        for node in self.nodes.values():
            text = json.dumps(node).casefold()
            node_terms = set(re.findall(r"[a-zA-Z_][\w.]*", text))
            overlap = len(terms & node_terms)
            score = overlap / max(1, len(terms))
            if score:
                scored.append((score, node))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        if not scored:
            return {"anchor": None, "similarity": 0.0, "pagerank": []}
        anchor_score, anchor = scored[0]
        ids = list(self.nodes)
        index = {node_id: i for i, node_id in enumerate(ids)}
        adjacency: list[set[int]] = [set() for _ in ids]
        for edge in self.edges:
            if edge["source"] in index and edge["target"] in index:
                adjacency[index[edge["source"]]].add(index[edge["target"]])
        n = len(ids)
        ranks = [1.0 / n] * n
        anchor_i = index[anchor["id"]]
        for _ in range(30):
            next_ranks = [0.15 / n] * n
            for i, outgoing in enumerate(adjacency):
                if not outgoing:
                    share = 0.85 * ranks[i] / n
                    next_ranks = [value + share for value in next_ranks]
                else:
                    share = 0.85 * ranks[i] / len(outgoing)
                    for target in outgoing:
                        next_ranks[target] += share
            next_ranks[anchor_i] += 0.70
            total = sum(next_ranks)
            ranks = [value / total for value in next_ranks]
        ranked = sorted(
            ((ranks[i], self.nodes[node_id]) for i, node_id in enumerate(ids)),
            key=lambda item: -item[0],
        )
        return {
            "anchor": anchor,
            "similarity": round(anchor_score, 4),
            "pagerank": [{"score": round(score, 6), "node": node} for score, node in ranked[:limit]],
        }

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
