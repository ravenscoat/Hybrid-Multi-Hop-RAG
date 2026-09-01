from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import load_chunks
from .config import Settings
from .index import HybridIndex
from .marker_convert import convert_pdfs
from .ollama import OllamaClient
from .pipeline import RAGPipeline
from .evaluation import evaluate_file, evaluate_retrieval_file


def ingest(path: Path, settings: Settings) -> None:
    chunks = load_chunks(path)
    if not chunks:
        raise SystemExit(f"No .txt, .md, or .pdf content found under {path}")
    ollama = OllamaClient(settings.ollama_url)
    embeddings: list[list[float]] = []
    batch_size = 16
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings.extend(ollama.embed(settings.embedding_model, [chunk.text for chunk in batch]))
        print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")
    HybridIndex(
        settings.vector_store_dir,
        settings.embedding_dim,
        settings.chroma_collection,
    ).build(chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks in {settings.vector_store_dir} (Chroma)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="hybrid-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("path", type=Path)
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("path", type=Path, help="folder containing PDFs")
    convert_parser.add_argument("--output", type=Path, default=Path("data/markdown"))
    convert_parser.add_argument("--mode", choices=["fast", "balanced"], default="fast")
    convert_parser.add_argument("--ocr", action="store_true", help="enable OCR")
    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    route = ask_parser.add_mutually_exclusive_group()
    route.add_argument(
        "--multi-hop", action="store_true", help="force the multi-hop pipeline"
    )
    route.add_argument(
        "--normal", action="store_true", help="force the fast single-hop pipeline"
    )
    ask_parser.add_argument("--rerank", action="store_true")
    eval_parser = subparsers.add_parser("eval", help="score a JSONL set with RAGAS")
    eval_parser.add_argument("input", type=Path, help="JSONL with question and optional reference")
    eval_parser.add_argument("--output", type=Path, default=Path("outputs/ragas.json"))
    eval_parser.add_argument("--rerank", action="store_true")
    eval_parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="run fast deterministic retrieval and reference-answer metrics only",
    )
    code_parser = subparsers.add_parser("code-agent", help="query the repository code knowledge graph")
    code_parser.add_argument("question")
    code_parser.add_argument("--knowledge-base", type=Path, default=Path("outputs/code_knowledge_base.json"))
    code_parser.add_argument("--semantic", action="store_true", help="use Qwen semantic anchors before PageRank")
    code_parser.add_argument("--embeddings", type=Path, default=Path("outputs/code_knowledge_base_embeddings.json"))
    eval_parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="score retrieval without Qwen answer generation",
    )
    args = parser.parse_args()
    if args.command == "code-agent":
        from .code_agent import CodeKnowledgeBase
        if not args.knowledge_base.exists():
            raise SystemExit(f"Knowledge base not found: {args.knowledge_base}. Run scripts/build_code_knowledge_base.py first.")
        kb = CodeKnowledgeBase(args.knowledge_base, args.embeddings if args.semantic else None)
        if args.semantic:
            if not args.embeddings.exists():
                raise SystemExit(f"Embeddings not found: {args.embeddings}. Run scripts/embed_code_knowledge_base.py first.")
            settings = Settings()
            query_text = f"Instruct: Retrieve code symbols relevant to this software-maintenance task.\nQuery: {args.question}"
            query_vector = OllamaClient(settings.ollama_url).embed(settings.embedding_model, [query_text])[0]
            result = kb.semantic_anchor_and_pagerank(query_vector, limit=15)
        else:
            result = kb.anchor_and_pagerank(args.question, limit=15)
        print(json.dumps(result, indent=2))
        return
    settings = Settings()
    if args.command == "convert":
        count = convert_pdfs(args.path, args.output, args.mode, disable_ocr=not args.ocr)
        print(f"Converted {count} PDF file(s) to Markdown in {args.output}")
    elif args.command == "ingest":
        ingest(args.path, settings)
    elif args.command == "eval":
        if args.retrieval_only:
            evaluate_retrieval_file(args.input, args.output, settings)
        else:
            evaluate_file(args.input, args.output, settings, args.rerank, not args.skip_ragas)
    else:
        selected_route = True if args.multi_hop else False if args.normal else None
        print(RAGPipeline(settings).answer(args.question, selected_route, args.rerank))


if __name__ == "__main__":
    main()
