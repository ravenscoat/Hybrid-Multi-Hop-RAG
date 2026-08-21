from __future__ import annotations

import argparse

from .config import Settings
from .pipeline import RAGPipeline


def build_app():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError('Gradio is not installed. Run: pip install -e ".[ui]"') from exc

    settings = Settings()

    def ask(question: str, route: str, rerank: bool):
        if not question.strip():
            return "Please enter a question.", "-", "-", ""
        try:
            selected_route = {"Automatic": None, "Normal": False, "Multi-hop": True}[route]
            pipeline = RAGPipeline(settings)
            answer, chosen_route, sources = pipeline.answer_with_trace(
                question.strip(), selected_route, rerank
            )
            source_text = "\n".join(f"- `{source}`" for source in sources) or "No chunks retrieved."
            debug = pipeline.last_rerank_debug
            if not debug.get("enabled"):
                debug_text = "Reranking disabled."
            else:
                debug_text = (
                    f"**Parse OK:** `{debug.get('parse_ok')}`  \n"
                    f"**Order changed:** `{debug.get('order_changed')}`  \n"
                    f"**Candidates:** `{debug.get('candidate_ids')}`  \n"
                    f"**Parsed IDs:** `{debug.get('parsed_ids')}`  \n"
                    f"**Final IDs:** `{debug.get('final_ids')}`  \n"
                    f"**Raw response:** `{debug.get('raw_response')}`"
                )
            return answer, chosen_route, source_text, debug_text
        except Exception as exc:
            return f"**RAG error:** `{type(exc).__name__}: {exc}`", "ERROR", "", ""

    with gr.Blocks(title="Local Hybrid RAG") as demo:
        gr.Markdown(
            "# Local Hybrid RAG\n\nAsk questions over your local Marker-processed documents. "
            "The app uses BM25 + HNSW + RRF, with optional adaptive multi-hop retrieval."
        )
        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="What does section 302 say about qatl-i-amd?",
                lines=3,
                scale=4,
            )
            with gr.Column(scale=1):
                route = gr.Radio(
                    ["Automatic", "Normal", "Multi-hop"], value="Automatic", label="Route"
                )
                rerank = gr.Checkbox(False, label="Use Qwen reranking")
                gr.Markdown("Reranking can take 30-60 seconds while Qwen loads.")
        ask_button = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")
        chosen = gr.Textbox(label="Chosen route", interactive=False)
        sources = gr.Markdown(label="Retrieved sources")
        rerank_debug = gr.Markdown(label="Reranker diagnostics")
        gr.Examples(
            examples=[
                ["What punishment is provided for qatl-i-amd under section 302?"],
                ["What is the definition of wrongful gain?"],
            ],
            inputs=question,
        )
        ask_button.click(ask, [question, route, rerank], [answer, chosen, sources, rerank_debug])
        question.submit(ask, [question, route, rerank], [answer, chosen, sources, rerank_debug])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(prog="hybrid-rag-ui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
