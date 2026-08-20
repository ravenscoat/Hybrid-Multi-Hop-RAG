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


SYSTEM_ANSWER = """Answer only from the supplied evidence. Each evidence block starts with an
exact citation label such as [pakistan_law.md#174]. Copy that exact label when
citing a claim; never write generic labels like [source#chunk]. If the evidence
is insufficient, say so plainly. Do not invent citations. Be concise."""
SYSTEM_ROUTER = """You route questions for a retrieval system. Choose MULTI_HOP only when
answering requires finding an intermediate entity or combining facts from sequential
retrieval steps. Otherwise choose SINGLE_HOP. Return JSON only."""


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ollama = OllamaClient(settings.ollama_url)
        self.index = HybridIndex(settings.index_dir, settings.embedding_dim)
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
            if (
                len(re.findall(r"\bsection\s+\d+\b", normalized)) >= 2
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
                )
                raw = self._record_llm_response(llm_span, response)
            multi_hop = self.parse_route(raw)
            route = "MULTI_HOP" if multi_hop else "SINGLE_HOP"
            span.set("rag.router.strategy", "qwen_llm")
            span.output(route)
            return multi_hop

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
            )
            raw = self._record_llm_response(span, response)
        try:
            values = json.loads(raw).get("subquestions", [])
            return [str(value) for value in values if str(value).strip()][:4]
        except (json.JSONDecodeError, AttributeError):
            return []

    def retrieve_multi_hop(self, question: str) -> list[SearchHit]:
        with rag_span("rag.multi_hop", {"input.value": question}, "CHAIN") as span:
            initial = self.retrieve(question)
            if not initial:
                span.output([])
                return []
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
                )
                answer = self._record_llm_response(generation_span, response)
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
