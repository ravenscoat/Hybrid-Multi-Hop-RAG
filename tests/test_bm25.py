from hybrid_rag.bm25 import BM25


def test_bm25_prefers_exact_term_match():
    index = BM25(["apples and pears", "neural semantic retrieval", "apples apples orchard"])
    hits = index.search("apples", limit=3)
    assert hits[0][0] == 2

