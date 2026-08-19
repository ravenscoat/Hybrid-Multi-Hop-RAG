from hybrid_rag.fusion import reciprocal_rank_fusion
from hybrid_rag.pipeline import RAGPipeline


def test_router_parses_valid_routes_and_fails_closed():
    assert RAGPipeline.parse_route('{"route":"MULTI_HOP"}') is True
    assert RAGPipeline.parse_route('{"route":"SINGLE_HOP"}') is False
    assert RAGPipeline.parse_route("not json") is False


def test_router_guardrail_detects_explicit_section_dependency():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    assert pipeline.decide_route(
        "According to Section 302, what proof is required by referenced Section 304?"
    ) is True


def test_rrf_rewards_documents_seen_by_both_retrievers():
    result = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]], constant=60)
    ids = [doc_id for doc_id, _ in result]
    assert ids[:2] == [1, 3]


def test_rrf_does_not_double_count_duplicate_in_one_ranking():
    result = reciprocal_rank_fusion([[1, 1], [2]], constant=60)
    scores = dict(result)
    assert scores[1] == scores[2]


def test_rrf_honors_limit():
    assert len(reciprocal_rank_fusion([[1, 2, 3]], limit=2)) == 2


def test_weighted_rrf_can_prefer_each_retriever():
    keyword_first = reciprocal_rank_fusion([[1, 2], [2, 1]], weights=[0.8, 0.2])
    semantic_first = reciprocal_rank_fusion([[1, 2], [2, 1]], weights=[0.2, 0.8])
    assert keyword_first[0][0] == 1
    assert semantic_first[0][0] == 2
