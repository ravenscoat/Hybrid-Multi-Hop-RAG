from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "into", "where", "which",
    "code", "locate", "change", "implementation", "should", "file", "function",
    "repository", "need", "needs", "make", "main", "when", "then", "also",
}


def _lexical_terms(value: str) -> set[str]:
    """Split prose, paths, dotted names, and snake_case identifiers."""
    normalized = re.sub(r"[_./\\:-]+", " ", value)
    return {
        term.casefold()
        for term in re.findall(r"[A-Za-z][A-Za-z0-9]*", normalized)
        if len(term) > 2 and term.casefold() not in STOPWORDS
    }


def _node_text(node: dict[str, Any]) -> str:
    return str(node.get("semantic_text") or " ".join(str(node.get(key, "")) for key in ("qualified_name", "name", "file", "path", "module", "signature", "docstring", "source_excerpt")))


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


class CodeKnowledgeBase:
    def __init__(self, path: Path, embeddings_path: Path | None = None):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.nodes = {item["id"]: item for item in payload.get("nodes", [])}
        self.edges = payload.get("edges", [])
        self.embeddings: dict[str, list[float]] = {}
        if embeddings_path and embeddings_path.exists():
            embedded = json.loads(embeddings_path.read_text(encoding="utf-8"))
            self.embeddings = embedded.get("embeddings", embedded)

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

    def anchor_and_pagerank(
        self, query: str, limit: int = 10, anchor_limit: int = 5
    ) -> dict[str, Any]:
        """Find IDF-weighted lexical anchors, then run personalized PageRank."""
        terms = _lexical_terms(query)
        if not terms:
            return {"anchor": None, "similarity": 0.0, "anchors": [], "pagerank": [], "anchor_method": "lexical_idf"}
        document_terms = {node_id: _lexical_terms(_node_text(node)) for node_id, node in self.nodes.items()}
        frequencies = Counter(term for node_terms in document_terms.values() for term in node_terms)
        total_nodes = max(1, len(self.nodes))
        scored: list[tuple[float, dict]] = []
        for node in self.nodes.values():
            node_terms = document_terms[node["id"]]
            overlap = terms & node_terms
            score = sum(math.log((total_nodes + 1) / (frequencies[term] + 1)) + 1 for term in overlap)
            score /= max(1.0, sum(math.log((total_nodes + 1) / (frequencies.get(term, 0) + 1)) + 1 for term in terms))
            if node.get("kind") == "function":
                score *= 1.15
            elif node.get("kind") == "import":
                score *= 0.8
            location = str(node.get("file") or node.get("path") or "")
            if location.startswith("tests/"):
                score *= 0.8
            elif location.startswith("scripts/"):
                score *= 0.85
            if score:
                scored.append((score, node))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        if not scored:
            return {"anchor": None, "similarity": 0.0, "anchors": [], "pagerank": [], "anchor_method": "lexical_idf"}
        return self._personalized_pagerank(scored[:anchor_limit], limit, "lexical_idf")

    def semantic_anchor_and_pagerank(
        self,
        query_vector: list[float],
        limit: int = 10,
        anchor_limit: int = 5,
    ) -> dict[str, Any]:
        """Seed PageRank with cached node embeddings and a query embedding."""
        if not self.embeddings:
            raise ValueError("Semantic embeddings are not loaded")
        scored: list[tuple[float, dict]] = []
        for node_id, node in self.nodes.items():
            vector = self.embeddings.get(node_id)
            if not vector:
                continue
            score = max(0.0, _cosine(query_vector, vector))
            if node.get("kind") == "function":
                score *= 1.08
            elif node.get("kind") == "import":
                score *= 0.8
            location = str(node.get("file") or node.get("path") or "")
            if location.startswith("tests/"):
                score *= 0.9
            if score:
                scored.append((score, node))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        if not scored:
            return {"anchor": None, "similarity": 0.0, "anchors": [], "pagerank": [], "anchor_method": "semantic"}
        return self._personalized_pagerank(scored[:anchor_limit], limit, "semantic")

    def _personalized_pagerank(
        self,
        anchors: list[tuple[float, dict]],
        limit: int,
        anchor_method: str,
    ) -> dict[str, Any]:
        anchor_score, anchor = anchors[0]
        ids = list(self.nodes)
        index = {node_id: i for i, node_id in enumerate(ids)}
        adjacency: list[list[tuple[int, float]]] = [[] for _ in ids]
        for edge in self.edges:
            if edge["source"] in index and edge["target"] in index:
                if edge.get("type") == "file_to_file":
                    weight = 1.0 + math.log1p(float(edge.get("cochange_count", 1) or 1))
                else:
                    weight = float(edge.get("weight", 1) or 1) * float(edge.get("confidence", 1) or 1)
                source_i, target_i = index[edge["source"]], index[edge["target"]]
                adjacency[source_i].append((target_i, weight))
                if edge.get("bidirectional") or edge.get("type") == "file_to_file":
                    adjacency[target_i].append((source_i, weight))
        n = len(ids)
        ranks = [1.0 / n] * n
        personalization = [0.0] * n
        total_similarity = sum(score for score, _ in anchors) or 1.0
        for score, node in anchors:
            personalization[index[node["id"]]] = score / total_similarity
        damping = 0.85
        for _ in range(40):
            next_ranks = [(1.0 - damping) * value for value in personalization]
            dangling = 0.0
            for i, outgoing in enumerate(adjacency):
                if not outgoing:
                    dangling += ranks[i]
                else:
                    total_weight = sum(weight for _, weight in outgoing)
                    for target, weight in outgoing:
                        next_ranks[target] += damping * ranks[i] * weight / total_weight
            for i, value in enumerate(personalization):
                next_ranks[i] += damping * dangling * value
            ranks = next_ranks
        ranked = sorted(
            ((ranks[i], self.nodes[node_id]) for i, node_id in enumerate(ids)),
            key=lambda item: -item[0],
        )
        return {
            "anchor": anchor,
            "similarity": round(anchor_score, 4),
            "anchor_method": anchor_method,
            "anchors": [
                {"similarity": round(score, 4), "node": node}
                for score, node in anchors
            ],
            "pagerank": [{"score": round(score, 6), "node": node} for score, node in ranked[:limit]],
        }

    def search(self, query: str, edge_type: str | None = None, limit: int = 12) -> dict[str, Any]:
        terms = _lexical_terms(query)
        matched = [node for node in self.nodes.values() if terms & _lexical_terms(json.dumps(node))]
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
