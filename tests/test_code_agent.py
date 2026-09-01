import json
from pathlib import Path
from unittest.mock import patch

from hybrid_rag.code_agent import CodeKnowledgeBase


def test_audit_and_personalized_pagerank():
    graph = {
        "nodes": [
            {"id": "function:retrieve", "kind": "function", "name": "retrieve"},
            {"id": "function:keyword", "kind": "function", "name": "keyword"},
            {"id": "file:pipeline.py", "kind": "file", "path": "pipeline.py"},
        ],
        "edges": [
            {"type": "function_to_function", "source": "function:retrieve", "target": "function:keyword"},
            {"type": "function_to_function", "source": "function:retrieve", "target": "function:keyword"},
        ],
    }
    with patch.object(Path, "read_text", return_value=json.dumps(graph)):
        kb = CodeKnowledgeBase(Path("code_agent_graph.json"))
    audit = kb.audit_and_deduplicate()
    result = kb.anchor_and_pagerank("retrieve pipeline", limit=3)
    assert audit["duplicate_edges_removed"] == 1
    anchor_ids = {item["node"]["id"] for item in result["anchors"]}
    assert "function:retrieve" in anchor_ids
    assert len(result["anchors"]) >= 1
    assert result["pagerank"][0]["score"] > 0


def test_anchor_tokenization_splits_snake_case_identifiers():
    graph = {
        "nodes": [
            {
                "id": "function:hybrid_rag.pipeline.parse_route",
                "kind": "function",
                "name": "parse_route",
                "file": "src/hybrid_rag/pipeline.py",
            }
        ],
        "edges": [],
    }
    with patch.object(Path, "read_text", return_value=json.dumps(graph)):
        kb = CodeKnowledgeBase(Path("code_agent_graph.json"))
    result = kb.anchor_and_pagerank("Where is the route decision parsed?")
    assert result["anchor"]["name"] == "parse_route"


def test_semantic_anchor_uses_bidirectional_cochange_and_containment():
    graph = {
        "nodes": [
            {"id": "file:a.py", "kind": "file", "path": "a.py"},
            {"id": "file:b.py", "kind": "file", "path": "b.py"},
            {"id": "function:b.target", "kind": "function", "name": "target", "file": "b.py"},
        ],
        "edges": [
            {"type": "file_to_file", "source": "file:a.py", "target": "file:b.py", "cochange_count": 3, "bidirectional": True},
            {"type": "file_contains_function", "source": "file:b.py", "target": "function:b.target", "weight": 2},
        ],
    }
    embeddings = {"embeddings": {"file:a.py": [1.0, 0.0], "file:b.py": [0.0, 1.0], "function:b.target": [0.0, 1.0]}}
    with patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", side_effect=[json.dumps(graph), json.dumps(embeddings)]):
        kb = CodeKnowledgeBase(Path("graph.json"), Path("embeddings.json"))
    result = kb.semantic_anchor_and_pagerank([1.0, 0.0], limit=3, anchor_limit=1)
    ranked_ids = [item["node"]["id"] for item in result["pagerank"]]
    assert result["anchor_method"] == "semantic"
    assert result["anchor"]["id"] == "file:a.py"
    assert "file:b.py" in ranked_ids
