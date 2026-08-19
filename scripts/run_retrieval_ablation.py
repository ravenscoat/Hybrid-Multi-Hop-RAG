from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from hybrid_rag.config import Settings
from hybrid_rag.evaluation import evaluate_file


CONFIGS = [
    ("bm25_50_semantic_50_no_ranker", 0.5, 0.5, False),
    ("bm25_50_semantic_50_qwen_ranker", 0.5, 0.5, True),
    ("bm25_60_semantic_40_no_ranker", 0.6, 0.4, False),
    ("bm25_60_semantic_40_qwen_ranker", 0.6, 0.4, True),
    ("bm25_40_semantic_60_no_ranker", 0.4, 0.6, False),
    ("bm25_40_semantic_60_qwen_ranker", 0.4, 0.6, True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare weighted hybrid-RAG retrieval settings")
    parser.add_argument("input", type=Path, help="JSONL evaluation set with reference_sources")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--skip-ragas", action="store_true", help="avoid slow local LLM judge calls")
    args = parser.parse_args()

    # Keep quality comparisons focused on retrieval/reranking. Phoenix tracing
    # can be enabled for a single chosen configuration after this sweep.
    os.environ["PHOENIX_ENABLED"] = "false"
    os.environ["MAX_CONTEXT_CHUNKS"] = "4"
    os.environ["RERANK_K"] = "4"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison: list[dict] = []
    base_settings = Settings()
    for name, bm25_weight, semantic_weight, rerank in CONFIGS:
        settings = replace(
            base_settings,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            max_context_chunks=4,
            rerank_k=4,
        )
        output = args.output_dir / f"{name}.json"
        evaluate_file(args.input, output, settings, rerank=rerank, run_ragas=not args.skip_ragas)
        report = json.loads(output.read_text(encoding="utf-8"))
        summary = {
            "configuration": name,
            "bm25_weight": bm25_weight,
            "semantic_weight": semantic_weight,
            "qwen_ranker": rerank,
            **report.get("summary", {}),
        }
        samples = report.get("samples", [])
        if samples:
            summary["latency_ms_mean"] = sum(float(s["latency_ms"]) for s in samples) / len(samples)
        comparison.append(summary)
        print(json.dumps(summary, sort_keys=True))
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
