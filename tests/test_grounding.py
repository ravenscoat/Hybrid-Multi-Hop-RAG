from hybrid_rag.models import Chunk
from hybrid_rag.pipeline import RAGPipeline


def test_document_hint_prefers_specific_title():
    assert RAGPipeline._document_hint("Section 146 of the Banking Companies Ordinance") == "banking companies ordinance"
    assert RAGPipeline._document_hint("ADBP Reorganization Ordinance section 5") == "agricultural development bank of pakistan"


def test_named_document_filters_rankings():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.index = type("Index", (), {})()
    pipeline.index.chunks = [
        Chunk(1, "Companies Ordinance, 1984\\law.md", "section 146", 455),
        Chunk(2, "Banking Companies Ordinance, 1962\\law.md", "section 146", 42),
    ]
    rankings, hint = pipeline._constrain_rankings(
        "What does Section 146 of the Companies Ordinance require?", [[2, 1], [2]]
    )
    assert hint == "companies ordinance"
    assert rankings == [[1], []]


def test_shortened_citation_is_repaired_when_unique():
    valid = {"Companies Act, 2017\\Companies Act, 2017.md#426"}
    answer = "The definition is stated here. [Companies Act, 2017.md#426]"
    repaired = RAGPipeline._repair_citations(answer, valid)
    assert repaired.endswith("[Companies Act, 2017\\Companies Act, 2017.md#426]")


def test_explicit_multiple_laws_route_to_multi_hop():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    question = "Compare Section 146 of the Companies Ordinance with Sections 4 and 5 of the ADBP Ordinance."
    assert pipeline.decide_route(question) is True


def test_explicit_legal_sections_get_targeted_subquestions():
    question = (
        "Under the Agricultural Development Bank of Pakistan Ordinance, compare "
        "Section 146 of the Companies Ordinance with Sections 4 and 5."
    )
    subquestions = RAGPipeline._deterministic_subquestions(question)
    assert any("Companies Ordinance" in value and "Section 146" in value for value in subquestions)
    assert any("Section 4" in value for value in subquestions)
    assert any("Section 5" in value for value in subquestions)


def test_mixed_named_laws_get_one_targeted_lookup_each():
    question = (
        "Compare the Abandoned Properties Act, the Access to the Media Act, "
        "the Agricultural Census Act, the Registration of Foreigners Act, "
        "and the Trained Paramedical Staff Facility Act."
    )
    hints = RAGPipeline._named_document_hints(question)
    assert len(hints) == 5
    assert len(RAGPipeline._deterministic_subquestions(question)) == 5
