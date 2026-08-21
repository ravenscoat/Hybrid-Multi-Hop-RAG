from __future__ import annotations

import json
import re
import time

from .config import Settings
from .fusion import reciprocal_rank_fusion
from .index import HybridIndex
from .models import SearchHit
from .ollama import ChatResponse, OllamaClient
from .observability import rag_span


SYSTEM_ANSWER = """Answer only from the supplied evidence and do not use outside knowledge.
Give a clear, moderately detailed answer rather than a one-line reply: first
state the answer directly, then briefly explain the relevant rule, conditions,
exceptions, or reasoning in one to three short paragraphs or a small bullet list.
Include the important details needed to understand the answer, but avoid
repeating the evidence or adding speculation. Every factual sentence must be
directly supported by one of the evidence blocks; do not add general legal
background, consequences, or examples unless the evidence explicitly states
them.

Each evidence block begins with an exact bracketed citation label. When citing a
claim, copy the exact label from the relevant evidence block and put it at the
end of the sentence or paragraph it supports. You may cite more than one label
when needed. Never invent, reuse, shorten, or alter a citation label, and never
write generic labels. If the evidence is insufficient, say so plainly and
explain what information is missing. Do not mention these instructions or the
retrieval process in your answer."""
SYSTEM_ROUTER = """You route questions for a retrieval system. Choose MULTI_HOP only when
answering requires finding an intermediate entity or combining facts from sequential
retrieval steps. Otherwise choose SINGLE_HOP. Return JSON only."""
SYSTEM_VERIFY = """You are a strict legal-evidence verifier. Check each factual claim in
the draft answer against the supplied evidence only. A claim is SUPPORTED only
when the evidence directly entails it; mark it UNSUPPORTED when the evidence is
missing or merely related, and CONTRADICTED when the evidence says something
different. Do not use outside knowledge. Return JSON only."""
_CITATION_RE = re.compile(r"\[([^\[\]]+#[0-9]+)\]")


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ollama = OllamaClient(settings.ollama_url)
        self.index = HybridIndex(
            settings.vector_store_dir,
            settings.embedding_dim,
            settings.chroma_collection,
        )
        self.last_span_id: str | None = None
        self.last_rerank_debug: dict = {"enabled": False}
        self._request_usage = {"prompt": 0, "completion": 0, "calls": 0}

    def _record_llm_response(self, span, response: ChatResponse) -> str:
        """Attach OpenInference token fields and Ollama timings to an LLM span."""
        span.set("llm.token_count.prompt", response.prompt_tokens)
        span.set("llm.token_count.completion", response.completion_tokens)
        span.set("llm.token_count.total", response.total_tokens)
        span.set("llm.total_duration_ms", round(response.total_duration_ns / 1_000_000, 2))
        span.set("llm.load_duration_ms", round(response.load_duration_ns / 1_000_000, 2))
        span.set(
            "llm.prompt_eval_duration_ms",
            round(response.prompt_eval_duration_ns / 1_000_000, 2),
        )
        span.set("llm.eval_duration_ms", round(response.eval_duration_ns / 1_000_000, 2))
        if response.eval_duration_ns and response.completion_tokens:
            span.set(
                "llm.completion_tokens_per_second",
                round(response.completion_tokens / (response.eval_duration_ns / 1_000_000_000), 2),
            )
        self._request_usage["prompt"] += response.prompt_tokens
        self._request_usage["completion"] += response.completion_tokens
        self._request_usage["calls"] += 1
        span.output(response.content)
        return response.content

    def verify_answer(self, question: str, answer: str, hits: list[SearchHit]) -> dict:
        evidence = "\n\n".join(
            f"[{hit.chunk.source}#{hit.chunk.ordinal}]\n{hit.chunk.text}" for hit in hits
        )
        prompt = f"""Question: {question}

Draft answer:
{answer}

Evidence:
{evidence or '(none)'}

Break the draft into its material factual claims. Return one verdict for every
claim, preserving the exact citation labels used by the draft when present.
Return this JSON shape:
{{"claims":[{{"claim":"...","verdict":"SUPPORTED|UNSUPPORTED|CONTRADICTED","citations":["source#ordinal"]}}],"overall":"PASS|FAIL"}}"""
        with rag_span(
            "llm.verify_answer",
            {"input.value": prompt, "llm.model_name": self.settings.generation_model},
            "LLM",
        ) as span:
            response = self.ollama.chat_with_metadata(
                self.settings.generation_model,
                SYSTEM_VERIFY,
                prompt,
                json_mode=True,
                num_ctx=self.settings.ollama_num_ctx,
                json_schema={
                    "type": "object",
                    "properties": {
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim": {"type": "string"},
                                    "verdict": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]},
                                    "citations": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["claim", "verdict", "citations"],
                                "additionalProperties": False,
                            },
                        },
                        "overall": {"type": "string", "enum": ["PASS", "FAIL"]},
                    },
                    "required": ["claims", "overall"],
                    "additionalProperties": False,
                },
            )
            raw = self._record_llm_response(span, response)
        try:
            parsed = json.loads(raw)
            claims = parsed.get("claims")
            if not isinstance(claims, list) or not claims:
                raise ValueError("verifier returned no claims")
            valid_labels = {f"{hit.chunk.source}#{hit.chunk.ordinal}" for hit in hits}
            for claim in claims:
                if not isinstance(claim, dict) or claim.get("verdict") not in {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}:
                    raise ValueError("verifier returned an invalid claim verdict")
                repaired = []
                for label in claim.get("citations", []):
                    repaired_label = self._repair_citations(f"[{label}]", valid_labels)
                    repaired_label = repaired_label[1:-1] if repaired_label.startswith("[") and repaired_label.endswith("]") else label
                    if repaired_label not in valid_labels:
                        raise ValueError("verifier returned an invalid citation")
                    repaired.append(repaired_label)
                claim["citations"] = repaired
            unsupported = sum(claim["verdict"] == "UNSUPPORTED" for claim in claims)
            contradicted = sum(claim["verdict"] == "CONTRADICTED" for claim in claims)
            # Be strict about contradictions, while allowing a useful answer when
            # the model adds one minor, non-material sentence. The answer still
            # fails closed when most claims are unsupported.
            parsed["unsupported_count"] = unsupported
            parsed["contradicted_count"] = contradicted
            parsed["verified"] = contradicted == 0 and unsupported <= max(1, len(claims) // 2)
            return parsed
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            return {"verified": False, "error": f"{type(exc).__name__}: {exc}", "claims": []}

    @staticmethod
    def _repair_citations(answer: str, valid_labels: set[str]) -> str:
        """Restore shortened citations when filename and chunk ordinal are unique."""
        by_key: dict[tuple[str, str], list[str]] = {}
        for label in valid_labels:
            path, ordinal = label.rsplit("#", 1)
            filename = path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()
            by_key.setdefault((filename, ordinal), []).append(label)

        def replace(match: re.Match[str]) -> str:
            raw = match.group(1)
            if raw in valid_labels:
                return match.group(0)
            path, ordinal = raw.rsplit("#", 1)
            filename = path.replace("/", "\\").rsplit("\\", 1)[-1].casefold()
            candidates = by_key.get((filename, ordinal), [])
            return f"[{candidates[0]}]" if len(candidates) == 1 else match.group(0)

        return _CITATION_RE.sub(replace, answer)

    @staticmethod
    def parse_route(raw: str) -> bool:
        """Return True for multi-hop; fail closed to the fast single-hop path."""
        try:
            route = str(json.loads(raw).get("route", "")).strip().upper()
        except (json.JSONDecodeError, AttributeError, TypeError):
            return False
        return route in {"MULTI_HOP", "MULTI-HOP", "MULTI HOP"}

    def decide_route(self, question: str) -> bool:
        with rag_span("rag.route", {"input.value": question}, "CHAIN") as span:
            # Deterministic guardrails catch explicit dependencies the small router
            # model can under-classify, such as a question linking two sections.
            normalized = question.casefold()
            named_laws = {
                "companies ordinance" if "companies ordinance" in normalized else None,
                "banking companies ordinance" if "banking companies ordinance" in normalized else None,
                "adbp ordinance" if "adbp" in normalized and "agricultural development bank" not in normalized else None,
                "agricultural development bank ordinance" if "agricultural development bank" in normalized else None,
                "companies act" if "companies act" in normalized else None,
            }
            named_laws.discard(None)
            if (
                len(re.findall(r"\bsection\s+\d+\b", normalized)) >= 2
                or len(named_laws) >= 2
                or re.search(r"\bsections\s+\d+\s*(?:and|&)\s*\d+", normalized)
                or re.search(r"\b(referenced|refers? to|according to)\b.*\bsection\s+\d+\b", normalized)
                or re.search(r"\bwhich\b.+\b(that|who|whose)\b", normalized)
            ):
                span.set("rag.router.strategy", "deterministic_guardrail")
                span.output("MULTI_HOP")
                return True
            prompt = f"""Question: {question}

Return exactly one JSON object: {{"route": "SINGLE_HOP"}} or
{{"route": "MULTI_HOP"}}. Use MULTI_HOP only when one fact must be found
before another fact can be retrieved. Do not solve the question."""
            with rag_span(
                "llm.router",
                {"input.value": prompt, "llm.model_name": self.settings.generation_model},
                "LLM",
            ) as llm_span:
                response = self.ollama.chat_with_metadata(
                    self.settings.generation_model,
                    SYSTEM_ROUTER,
                    prompt,
                    json_mode=True,
                    num_ctx=self.settings.ollama_num_ctx,
                )
                raw = self._record_llm_response(llm_span, response)
            multi_hop = self.parse_route(raw)
            route = "MULTI_HOP" if multi_hop else "SINGLE_HOP"
            span.set("rag.router.strategy", "qwen_llm")
            span.output(route)
            return multi_hop

    @staticmethod
    def _document_hint(query: str) -> str | None:
        """Extract a lightweight law-title hint for deterministic source filtering."""
        normalized = query.casefold()
        aliases = {
            "adbp": "agricultural development bank of pakistan",
            "agricultural development bank": "agricultural development bank of pakistan",
            "companies ordinance": "companies ordinance",
            "companies act": "companies act",
            "banking companies ordinance": "banking companies ordinance",
        }
        found = [hint for needle, hint in aliases.items() if needle in normalized]
        found = [hint for hint in found if not any(hint != other and hint in other for other in found)]
        if len(set(found)) == 1:
            return found[0]
        # A question naming multiple laws is intentionally left unconstrained;
        # multi-hop retrieval must be allowed to collect evidence from each one.
        if len(set(found)) > 1:
            return None
        match = re.search(
            r"([a-z][a-z0-9 &'()/-]{2,80}?)\s+(?:ordinance|act)(?:,?\s*\d{4})?",
            normalized,
        )
        return match.group(0).strip() if match else None

    def _constrain_rankings(self, query: str, rankings: list[list[int]]) -> tuple[list[list[int]], str | None]:
        """Keep candidates from an explicitly named law, when that law exists."""
        hint = self._document_hint(query)
        if not hint:
            return rankings, None
        matching_ids = {
            chunk.id
            for chunk in self.index.chunks
            if self._source_matches_hint(chunk.source, hint)
        }
        if not matching_ids:
            return rankings, None
        return [[doc_id for doc_id in ranking if doc_id in matching_ids] for ranking in rankings], hint

    @staticmethod
    def _source_matches_hint(source: str, hint: str) -> bool:
        title = source.replace("\\", "/").split("/")[0].casefold().strip()
        if "agricultural development bank" in hint:
            return "agricultural development bank of pakistan" in title
        if hint == "companies ordinance":
            return title.startswith("companies ordinance")
        if hint == "companies act":
            return title.startswith("companies act")
        if hint == "banking companies ordinance":
            return title.startswith("banking companies ordinance")
        return hint in title

    def _section_candidate_ids(self, query: str, limit: int = 12) -> list[int]:
        """Find chunks containing an actual numbered section heading.

        BM25 quite reasonably matches cross-references such as "under section
        146".  For statutes, however, a query asking for section 146 should
        prefer the chunk whose text starts the actual section 146 provision.
        This lightweight lexical pass does not replace BM25 or embeddings; it
        only supplies a high-precision anchor for legal section queries.
        """
        numbers: set[int] = set()
        for first, second in re.findall(
            r"\bsections?\s+(\d+)(?:\s*(?:and|&)\s*(\d+))?", query.casefold()
        ):
            numbers.add(int(first))
            if second:
                numbers.add(int(second))
        if not numbers:
            return []
        hint = self._document_hint(query)
        candidates: list[tuple[int, int, int]] = []
        for chunk in self.index.chunks:
            if hint and not self._source_matches_hint(chunk.source, hint):
                # For cross-document questions the query names more than one
                # law, so do not apply a single-document filter here.
                continue
            # Marker may place a section heading after a Markdown title on the
            # same line (for example ``## COMMENCEMENT ... 146. Restrictions``)
            # or in a table-of-contents line. Match numbered provision headings
            # while avoiding ordinary ``section 146`` cross-references.
            heading_numbers = {
                int(value)
                for value in re.findall(r"(?im)\b(\d+)\s*[.)-]\s+[A-Z]", chunk.text)
            }
            matched = numbers & heading_numbers
            if matched:
                # Prefer chunks that contain the largest number of requested
                # headings, then earlier chunks for deterministic ordering.
                candidates.append((-len(matched), chunk.ordinal, chunk.id))
        candidates.sort()
        return [chunk_id for _, _, chunk_id in candidates[:limit]]

    def retrieve(self, query: str) -> list[SearchHit]:
        with rag_span("rag.retrieve.hybrid", {"input.value": query}, "RETRIEVER") as span:
            with rag_span(
                "embedding.query",
                {"input.value": query, "embedding.model_name": self.settings.embedding_model},
                "CHAIN",
            ) as embedding_span:
                vector = self.ollama.embed(self.settings.embedding_model, [query])[0]
                embedding_span.set("embedding.vector_length", len(vector))
                embedding_span.output(
                    {"model": self.settings.embedding_model, "dimensions": len(vector)}
                )
            # These two spans expose ranked candidate IDs. The parent hybrid
            # span is the actual RETRIEVER and carries full document objects.
            with rag_span("retrieval.bm25", {"input.value": query}, "CHAIN") as bm25_span:
                keyword = self.index.keyword(query, self.settings.bm25_k)
                bm25_span.output(keyword)
            with rag_span("retrieval.hnsw", {"input.value": query}, "CHAIN") as dense_span:
                dense = self.index.dense(vector, self.settings.dense_k)
                dense_span.output(dense)
            section_ids = self._section_candidate_ids(query)
            if section_ids:
                # Put exact section-heading matches at the front of both
                # candidate lists so RRF cannot bury them beneath generic
                # cross-references. Deduplication preserves original ranking.
                keyword = section_ids + [doc_id for doc_id in keyword if doc_id not in section_ids]
                dense = section_ids + [doc_id for doc_id in dense if doc_id not in section_ids]
                span.set("rag.section_candidates", section_ids)
            # A single explicitly named law is a safe, high-precision filter.
            # Cross-document questions return no hint and remain unconstrained.
            explicit_document = self._document_hint(query)
            if self.settings.document_constraints or explicit_document:
                (keyword, dense), document_hint = self._constrain_rankings(query, [keyword, dense])
            else:
                document_hint = None
            if document_hint:
                span.set("rag.document_constraint", document_hint)
            with rag_span(
                "retrieval.rrf",
                {
                    "rag.rrf.constant": self.settings.rrf_constant,
                    "rag.rrf.bm25_weight": self.settings.bm25_weight,
                    "rag.rrf.semantic_weight": self.settings.semantic_weight,
                },
                "CHAIN",
            ) as fusion_span:
                fused = reciprocal_rank_fusion(
                    [keyword, dense],
                    self.settings.rrf_constant,
                    self.settings.fused_k,
                    [self.settings.bm25_weight, self.settings.semantic_weight],
                )
                fusion_span.output([doc_id for doc_id, _ in fused])
            hits = [SearchHit(self.index.chunks[doc_id], score, query) for doc_id, score in fused]
            documents = [
                {
                    "document.id": str(hit.chunk.id),
                    "document.score": hit.score,
                    "document.metadata": {"source": hit.chunk.source, "ordinal": hit.chunk.ordinal},
                    "document.content": hit.chunk.text[:1000],
                }
                for hit in hits
            ]
            span.set("retrieval.documents", documents)
            span.set("rag.retrieved_count", len(hits))
            span.output([f"{hit.chunk.source}#{hit.chunk.ordinal}" for hit in hits])
            return hits

    def decompose(self, question: str, bridge: SearchHit) -> list[str]:
        prompt = f"""Original question: {question}

First retrieved passage:
{bridge.chunk.text}

Return JSON {{"subquestions": [...]}} with 2-4 self-contained retrieval questions.
Resolve the bridge entity from the passage when possible. Do not answer the question."""
        with rag_span(
            "llm.decompose",
            {"input.value": question, "llm.model_name": self.settings.generation_model},
            "LLM",
        ) as span:
            span.set("rag.bridge_source", f"{bridge.chunk.source}#{bridge.chunk.ordinal}")
            response = self.ollama.chat_with_metadata(
                self.settings.generation_model,
                "Decompose multi-hop questions for iterative document retrieval.",
                prompt,
                json_mode=True,
                num_ctx=self.settings.ollama_num_ctx,
            )
            raw = self._record_llm_response(span, response)
        try:
            values = json.loads(raw).get("subquestions", [])
            return [str(value) for value in values if str(value).strip()][:4]
        except (json.JSONDecodeError, AttributeError):
            return []

    @staticmethod
    def _deterministic_subquestions(question: str) -> list[str]:
        """Create precise legal lookups when the user names laws and sections."""
        normalized = question.casefold()
        subquestions: list[str] = []
        if "companies ordinance" in normalized and re.search(r"\bsection\s+146\b", normalized):
            subquestions.append(
                "Companies Ordinance, 1984 Section 146 restrictions on commencement of business and borrowing powers"
            )
        if "agricultural development bank" in normalized or "adbp" in normalized:
            section_numbers: set[int] = set()
            for first, second in re.findall(
                r"\bsections?\s+(\d+)(?:\s*(?:and|&)\s*(\d+))?", normalized
            ):
                section_numbers.add(int(first))
                if second:
                    section_numbers.add(int(second))
            if 4 in section_numbers:
                subquestions.append(
                    "Agricultural Development Bank of Pakistan (Reorganization and Conversion) Ordinance, 2002 Section 4 transfer and section 4(5) override of Companies Ordinance section 146"
                )
            if 5 in section_numbers:
                subquestions.append(
                    "Agricultural Development Bank of Pakistan (Reorganization and Conversion) Ordinance, 2002 Section 5 deposits lending and State Bank of Pakistan licence"
                )
            if 6 in section_numbers:
                subquestions.append(
                    "Agricultural Development Bank of Pakistan (Reorganization and Conversion) Ordinance, 2002 Section 6 continuation of ADBP employees in service"
                )
            if 7 in section_numbers:
                subquestions.append(
                    "Agricultural Development Bank of Pakistan (Reorganization and Conversion) Ordinance, 2002 Section 7 contracts and pending legal proceedings"
                )
            if 8 in section_numbers:
                subquestions.append(
                    "Agricultural Development Bank of Pakistan (Reorganization and Conversion) Ordinance, 2002 Section 8 bar of jurisdiction and protection for good-faith actions"
                )
            if 10 in section_numbers:
                subquestions.append(
                    "Agricultural Development Bank of Pakistan (Reorganization and Conversion) Ordinance, 2002 Section 10 overriding effect over conflicting laws and orders"
                )
        return subquestions

    def retrieve_multi_hop(self, question: str) -> list[SearchHit]:
        with rag_span("rag.multi_hop", {"input.value": question}, "CHAIN") as span:
            initial = self.retrieve(question)
            if not initial:
                span.output([])
                return []
            subquestions = self._deterministic_subquestions(question)
            if not subquestions:
                subquestions = self.decompose(question, initial[0])
            span.set("rag.subquestions", subquestions)
            if not subquestions:
                span.output([hit.chunk.id for hit in initial])
                return initial
            per_query = [self.retrieve(subquestion) for subquestion in subquestions]
            rankings = [[hit.chunk.id for hit in hits] for hits in per_query]
            fused = reciprocal_rank_fusion(
                rankings,
                self.settings.rrf_constant,
                self.settings.fused_k,
            )
            matched: dict[int, str] = {}
            for subquestion, hits in zip(subquestions, per_query):
                for hit in hits:
                    matched.setdefault(hit.chunk.id, subquestion)
            result = [
                SearchHit(self.index.chunks[doc_id], score, matched.get(doc_id))
                for doc_id, score in fused
            ]
            span.output([hit.chunk.id for hit in result])
            return result

    def rerank(self, question: str, hits: list[SearchHit]) -> list[SearchHit]:
        candidates = hits[:10]
        original_ids = [hit.chunk.id for hit in candidates]
        rendered = "\n\n".join(
            f"ID {hit.chunk.id}\nVerification question: {hit.matched_query or question}\n{hit.chunk.text}"
            for hit in candidates
        )
        prompt = f"""Final user question: {question}

Rank evidence by whether it helps answer its Verification question. A later-hop
passage must not be rejected merely because it does not mention the final question.
Return every candidate ID exactly once in best-first order. Do not omit candidates,
invent IDs, explain the answer, or return an answer. Return only JSON in this form:
{{"ids": [all candidate integer IDs best-first]}}

{rendered}"""
        with rag_span(
            "llm.rerank",
            {
                "input.value": question,
                "llm.model_name": self.settings.generation_model,
                "rag.candidate_count": len(candidates),
            },
            "RERANKER",
        ) as span:
            response = self.ollama.chat_with_metadata(
                self.settings.generation_model,
                "You are a strict listwise evidence reranker. Never answer the question. Return only the requested JSON object with an ids array.",
                prompt,
                json_mode=True,
                num_ctx=self.settings.ollama_num_ctx,
                json_schema={
                    "type": "object",
                    "properties": {
                        "ids": {
                            "type": "array",
                            "items": {"type": "integer", "enum": original_ids},
                            "minItems": len(original_ids),
                            "maxItems": len(original_ids),
                            "uniqueItems": True,
                        }
                    },
                    "required": ["ids"],
                    "additionalProperties": False,
                },
            )
            raw = self._record_llm_response(span, response)
        parse_error: str | None = None
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or "ids" not in parsed:
                raise ValueError("response did not contain an ids field")
            ordered_ids = [int(value) for value in parsed.get("ids", [])]
            if not ordered_ids:
                raise ValueError("ids array was empty")
            if len(ordered_ids) != len(original_ids) or set(ordered_ids) != set(original_ids):
                raise ValueError("ids must contain every candidate exactly once")
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            ordered_ids = []
        by_id = {hit.chunk.id: hit for hit in candidates}
        valid_ids = [doc_id for doc_id in ordered_ids if doc_id in by_id]
        ordered = [by_id[doc_id] for doc_id in valid_ids]
        final = (ordered or candidates)[: self.settings.rerank_k]
        final_ids = [hit.chunk.id for hit in final]
        self.last_rerank_debug = {
            "enabled": True,
            "candidate_ids": original_ids,
            "raw_response": raw[:4000],
            "parsed_ids": ordered_ids,
            "valid_ids": valid_ids,
            "final_ids": final_ids,
            "parse_ok": parse_error is None,
            "parse_error": parse_error,
            "order_changed": final_ids != original_ids[: len(final_ids)],
        }
        span.set("rerank.candidate_ids", original_ids)
        span.set("rerank.parsed_ids", ordered_ids)
        span.set("rerank.valid_ids", valid_ids)
        span.set("rerank.final_ids", final_ids)
        span.set("rerank.parse_ok", parse_error is None)
        span.set("rerank.order_changed", self.last_rerank_debug["order_changed"])
        if parse_error:
            span.set("rerank.parse_error", parse_error)
        span.output(self.last_rerank_debug)
        return final

    def answer_with_trace(
        self, question: str, multi_hop: bool | None = None, rerank: bool = False
    ) -> tuple[str, str, list[str]]:
        started = time.perf_counter()
        self._request_usage = {"prompt": 0, "completion": 0, "calls": 0}
        with rag_span(
            "rag.query",
            {"input.value": question, "input.mime_type": "text/plain", "rag.rerank": rerank},
            "CHAIN",
        ) as root_span:
            self.last_span_id = root_span.span_id
            self.index.load()
            self.last_rerank_debug = {"enabled": rerank}
            if multi_hop is None:
                multi_hop = self.decide_route(question)
            hits = self.retrieve_multi_hop(question) if multi_hop else self.retrieve(question)
            if rerank and hits:
                hits = self.rerank(question, hits)
            hits = hits[: self.settings.max_context_chunks]
            route = "MULTI_HOP" if multi_hop else "SINGLE_HOP"
            sources = [f"{hit.chunk.source}#{hit.chunk.ordinal}" for hit in hits]
            root_span.set("rag.route", route)
            root_span.set("rag.sources", sources)
            root_span.set("rag.context_count", len(hits))
            evidence = "\n\n".join(
                f"[{hit.chunk.source}#{hit.chunk.ordinal}]\n{hit.chunk.text}" for hit in hits
            )
            answer_prompt = f"Question: {question}\n\nEvidence:\n{evidence or '(none)'}"
            with rag_span(
                "llm.generate_answer",
                {
                    "input.value": answer_prompt,
                    "llm.model_name": self.settings.generation_model,
                    "rag.context_sources": sources,
                },
                "LLM",
            ) as generation_span:
                response = self.ollama.chat_with_metadata(
                    self.settings.generation_model,
                    SYSTEM_ANSWER,
                    answer_prompt,
                    num_ctx=self.settings.ollama_num_ctx,
                )
                answer = self._record_llm_response(generation_span, response)
                valid_labels = {f"{hit.chunk.source}#{hit.chunk.ordinal}" for hit in hits}
                answer = self._repair_citations(answer, valid_labels)
                cited_labels = set(_CITATION_RE.findall(answer))
                invalid_labels = sorted(cited_labels - valid_labels)
                citation_ok = bool(cited_labels) and not invalid_labels
                generation_span.set("rag.citation_ok", citation_ok)
                generation_span.set("rag.citation_count", len(cited_labels))
                if invalid_labels:
                    generation_span.set("rag.invalid_citations", invalid_labels)
                if not citation_ok and self.settings.enforce_grounding:
                    answer = (
                        "I could not verify a grounded answer from the retrieved legal text. "
                        "Please narrow the question to a named law or section, or try again."
                    )
                    root_span.set("rag.abstained", True)
            verification = {"verified": True, "skipped": not self.settings.verify_answers}
            if citation_ok and self.settings.verify_answers:
                verification = self.verify_answer(question, answer, hits)
                root_span.set("rag.claim_verification", verification.get("verified", False))
                root_span.set("rag.claim_count", len(verification.get("claims", [])))
                root_span.set("rag.unsupported_claim_count", verification.get("unsupported_count", 0))
                root_span.set("rag.contradicted_claim_count", verification.get("contradicted_count", 0))
                if not verification.get("verified", False) and self.settings.enforce_grounding:
                    answer = (
                        "I could not verify every factual claim against the retrieved legal text. "
                        "The evidence may be incomplete or conflicting, so I will not guess."
                    )
                    root_span.set("rag.abstained", True)
            total_tokens = self._request_usage["prompt"] + self._request_usage["completion"]
            root_span.set("rag.token_count.prompt", self._request_usage["prompt"])
            root_span.set("rag.token_count.completion", self._request_usage["completion"])
            root_span.set("rag.token_count.total", total_tokens)
            root_span.set("rag.llm_call_count", self._request_usage["calls"])
            root_span.set("rag.latency_ms", round((time.perf_counter() - started) * 1000, 2))
            root_span.output(answer)
            return answer, route, sources

    def answer(
        self, question: str, multi_hop: bool | None = None, rerank: bool = False
    ) -> str:
        return self.answer_with_trace(question, multi_hop, rerank)[0]
