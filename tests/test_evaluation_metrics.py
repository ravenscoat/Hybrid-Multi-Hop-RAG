from hybrid_rag.evaluation import _retrieval_metrics
from hybrid_rag.ollama import ChatResponse


def test_retrieval_metrics_rank_relevant_chunks():
    scores = _retrieval_metrics(
        ["law.md#9", "law.md#2", "law.md#5"],
        ["law.md#2", "law.md#5"],
    )

    assert scores["retrieval_precision"] == 2 / 3
    assert scores["retrieval_recall"] == 1.0
    assert scores["retrieval_mrr"] == 0.5
    assert scores["retrieval_hit_rate"] == 1.0
    assert 0.0 < scores["retrieval_ndcg"] < 1.0


def test_document_level_reference_matches_any_chunk():
    scores = _retrieval_metrics(["folder\\law.md#22"], ["folder/law.md"])
    assert scores["retrieval_recall"] == 1.0


def test_ollama_chat_response_exposes_usage():
    response = ChatResponse.from_ollama(
        {
            "message": {"content": "answer"},
            "prompt_eval_count": 120,
            "eval_count": 30,
            "total_duration": 1_500_000,
        }
    )

    assert response.content == "answer"
    assert response.prompt_tokens == 120
    assert response.completion_tokens == 30
    assert response.total_tokens == 150
