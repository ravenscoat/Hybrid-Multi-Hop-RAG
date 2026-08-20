from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from statistics import fmean

from .config import Settings
from .pipeline import RAGPipeline


def evaluate_file(
    input_path: Path,
    output_path: Path,
    settings: Settings,
    rerank: bool = False,
    run_ragas: bool = True,
) -> None:
    """Run RAGAS faithfulness and response relevancy on a JSONL test set.

    Each line must contain ``question`` and may contain a gold ``reference``
    answer and ``reference_sources`` list. The latter enables deterministic
    retrieval precision, recall, F1, hit-rate, MRR, and NDCG.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError('Install evaluation dependencies with: pip install -e ".[eval]"') from exc

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pipeline = RAGPipeline(settings)
    samples: list[dict] = []
    for row in rows:
        question = str(row["question"])
        started = time.perf_counter()
        answer, route, sources = pipeline.answer_with_trace(question, rerank=rerank)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        # Re-run retrieval to expose the exact context text to RAGAS.
        hits = pipeline.retrieve_multi_hop(question) if route == "MULTI_HOP" else pipeline.retrieve(question)
        samples.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": [hit.chunk.text for hit in hits[: settings.max_context_chunks]],
            **({"reference": row["reference"]} if row.get("reference") else {}),
            **(
                {"reference_sources": [str(value) for value in row["reference_sources"]]}
                if row.get("reference_sources")
                else {}
            ),
            "route": route,
            "sources": sources,
            "phoenix_span_id": pipeline.last_span_id,
            "latency_ms": latency_ms,
            "rerank_debug": pipeline.last_rerank_debug,
        })

    score_rows: list[dict]
    if run_ragas:
        try:
            from ragas import EvaluationDataset, RunConfig, evaluate
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import Faithfulness, ResponseRelevancy
            from langchain_ollama import ChatOllama, OllamaEmbeddings
        except ImportError as exc:
            raise RuntimeError('Install evaluation dependencies with: pip install -e ".[eval]"') from exc
        judge = LangchainLLMWrapper(
            ChatOllama(
                model=settings.generation_model,
                base_url=settings.ollama_url,
                temperature=0,
                reasoning=False,
                num_predict=1024,
                keep_alive="10m",
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_url)
        )
        metrics = [Faithfulness(llm=judge), ResponseRelevancy(llm=judge, embeddings=embeddings)]
        ragas_rows = [
            {
                key: value
                for key, value in sample.items()
                if key in {"user_input", "response", "retrieved_contexts", "reference"}
            }
            for sample in samples
        ]
        # Local models can take a long time to produce judge JSON. Keep each
        # metric call bounded and avoid retry storms; callers can increase
        # these limits with environment variables for a larger evaluation.
        ragas_timeout = int(os.getenv("RAGAS_TIMEOUT_SECONDS", "60"))
        ragas_retries = int(os.getenv("RAGAS_MAX_RETRIES", "0"))
        result = evaluate(
            EvaluationDataset.from_list(ragas_rows),
            metrics=metrics,
            run_config=RunConfig(
                timeout=ragas_timeout,
                max_retries=ragas_retries,
                max_workers=1,
            ),
        )
        score_rows = result.to_pandas().to_dict(orient="records")
        for score in score_rows:
            if "answer_relevancy" in score:
                score["response_relevancy"] = score.pop("answer_relevancy")
    else:
        score_rows = [{} for _ in samples]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for sample, score in zip(samples, score_rows):
        score.update(_retrieval_metrics(sample.get("sources", []), sample.get("reference_sources")))
        if sample.get("reference"):
            score.update(_answer_metrics(sample["response"], sample["reference"], scorer))
    summary = _summarize_scores(score_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "scores": score_rows, "samples": samples}, indent=2, default=str),
        encoding="utf-8",
    )
    _log_scores_to_phoenix(samples, score_rows)
    print(f"RAGAS evaluation written to {output_path}")


def _log_scores_to_phoenix(samples: list[dict], scores: list[dict]) -> None:
    if os.getenv("PHOENIX_ENABLED", "false").casefold() != "true":
        return
    try:
        from phoenix.client import Client
    except ImportError:
        return
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006")
    client = Client(base_url=endpoint)
    for sample, score_row in zip(samples, scores):
        span_id = sample.get("phoenix_span_id")
        if not span_id:
            continue
        metric_config = {
            "faithfulness": ("LLM", 0.7, "ragas"),
            "response_relevancy": ("LLM", 0.7, "ragas"),
            "retrieval_precision": ("CODE", 0.7, "deterministic"),
            "retrieval_recall": ("CODE", 0.8, "deterministic"),
            "retrieval_f1": ("CODE", 0.7, "deterministic"),
            "retrieval_hit_rate": ("CODE", 1.0, "deterministic"),
            "retrieval_mrr": ("CODE", 0.7, "deterministic"),
            "retrieval_ndcg": ("CODE", 0.7, "deterministic"),
            "rouge1_f1": ("CODE", 0.5, "rouge-score"),
            "rouge2_f1": ("CODE", 0.3, "rouge-score"),
            "rougeL_f1": ("CODE", 0.4, "rouge-score"),
            "answer_token_precision": ("CODE", 0.6, "deterministic"),
            "answer_token_recall": ("CODE", 0.6, "deterministic"),
            "answer_token_f1": ("CODE", 0.6, "deterministic"),
            "answer_exact_match": ("CODE", 1.0, "deterministic"),
        }
        for metric, (kind, threshold, framework) in metric_config.items():
            value = score_row.get(metric)
            if value is None or not math.isfinite(float(value)):
                continue
            client.spans.add_span_annotation(
                span_id=span_id,
                annotation_name=metric,
                annotator_kind=kind,
                score=float(value),
                label="pass" if float(value) >= threshold else "review",
                explanation=f"Local {framework} {metric} score.",
                metadata={"framework": framework, "threshold": threshold},
                sync=True,
            )


def _source_matches(retrieved: str, relevant: str) -> bool:
    retrieved = retrieved.replace("/", "\\").casefold().strip()
    relevant = relevant.replace("/", "\\").casefold().strip()
    return retrieved == relevant or ("#" not in relevant and retrieved.startswith(relevant + "#"))


def _retrieval_metrics(retrieved: list[str], relevant: list[str] | None) -> dict[str, float]:
    if not relevant:
        return {}
    flags = [any(_source_matches(item, gold) for gold in relevant) for item in retrieved]
    # Precision/recall are measured over retrieved chunks, but NDCG is
    # source-based here.  Multiple chunks from the same relevant document
    # must not receive repeated gain; otherwise NDCG can incorrectly exceed
    # its maximum value of 1.0.
    seen_relevant: set[str] = set()
    unique_flags: list[bool] = []
    for item, matched in zip(retrieved, flags):
        if not matched:
            unique_flags.append(False)
            continue
        matched_gold = next(
            gold for gold in relevant if _source_matches(item, gold)
        )
        key = matched_gold.replace("/", "\\").casefold().strip()
        unique_flags.append(key not in seen_relevant)
        seen_relevant.add(key)
    matched_gold = sum(any(_source_matches(item, gold) for item in retrieved) for gold in relevant)
    relevant_retrieved = sum(flags)
    precision = relevant_retrieved / len(retrieved) if retrieved else 0.0
    recall = matched_gold / len(relevant)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    first_rank = next((rank for rank, matched in enumerate(flags, 1) if matched), None)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, matched in enumerate(unique_flags, 1)
        if matched
    )
    ideal_count = min(len(relevant), len(retrieved))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "retrieval_f1": f1,
        "retrieval_hit_rate": float(bool(relevant_retrieved)),
        "retrieval_mrr": 1.0 / first_rank if first_rank else 0.0,
        "retrieval_ndcg": dcg / idcg if idcg else 0.0,
    }


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold())


def _answer_metrics(response: str, reference: str, scorer) -> dict[str, float]:
    rouge = scorer.score(reference, response)
    predicted = Counter(_tokens(response))
    gold = Counter(_tokens(reference))
    overlap = sum((predicted & gold).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(gold.values()) if gold else 0.0
    token_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rouge1_f1": rouge["rouge1"].fmeasure,
        "rouge2_f1": rouge["rouge2"].fmeasure,
        "rougeL_f1": rouge["rougeL"].fmeasure,
        "answer_token_precision": precision,
        "answer_token_recall": recall,
        "answer_token_f1": token_f1,
        "answer_exact_match": float(" ".join(_tokens(response)) == " ".join(_tokens(reference))),
    }


def _summarize_scores(rows: list[dict]) -> dict[str, float]:
    names = {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return {
        name: fmean(float(row[name]) for row in rows if name in row and math.isfinite(float(row[name])))
        for name in sorted(names)
        if any(name in row and math.isfinite(float(row[name])) for row in rows)
    }
