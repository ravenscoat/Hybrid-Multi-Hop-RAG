from __future__ import annotations

from pathlib import Path
import textwrap

import cv2
import numpy as np


W, H = 1920, 1080
BG = (16, 16, 18)
PANEL = (35, 35, 39)
TEXT = (240, 240, 244)
MUTED = (170, 170, 180)
ORANGE = (30, 112, 245)
GREEN = (50, 220, 115)
PURPLE = (180, 100, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, xy, size=0.8, color=TEXT, thick=1, line=34):
    x, y = xy
    for row in text.split("\n"):
        cv2.putText(img, row, (x, y), FONT, size, color, thick, cv2.LINE_AA)
        y += line
    return y


def wrapped(img, text, xy, width=88, size=0.65, color=TEXT, line=30):
    rows = []
    for para in text.split("\n"):
        rows.extend(textwrap.wrap(para, width=width) or [""])
    return put(img, "\n".join(rows), xy, size=size, color=color, line=line)


def base(kicker: str, title: str):
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (W, 8), ORANGE, -1)
    put(img, kicker.upper(), (80, 80), 0.55, ORANGE, 2)
    put(img, title, (80, 145), 1.25, TEXT, 2, 48)
    put(img, "LOCAL HYBRID RAG  •  BM25 + SEMANTIC + RRF  •  QWEN3", (80, 1015), 0.48, MUTED, 1)
    return img


def frame_query():
    img = base("01  /  Ask", "Ask a question over 994 legal documents")
    cv2.rectangle(img, (80, 215), (1840, 360), PANEL, -1)
    put(img, "QUESTION", (110, 260), 0.48, MUTED, 1)
    wrapped(img, "What punishment is provided for qatl-i-amd under section 302?", (110, 315), 105, 0.82)
    cv2.rectangle(img, (80, 425), (1840, 860), PANEL, -1)
    put(img, "PIPELINE", (110, 475), 0.5, MUTED, 1)
    steps = [("1", "Adaptive route", "SINGLE_HOP", GREEN), ("2", "Keyword search", "BM25", ORANGE), ("3", "Semantic search", "Chroma / HNSW", PURPLE), ("4", "Fusion", "Reciprocal Rank Fusion", GREEN)]
    x = 120
    for n, a, b, c in steps:
        cv2.circle(img, (x, 590), 34, c, -1)
        put(img, n, (x - 9, 600), 0.75, BG, 2)
        put(img, a, (x - 60, 680), 0.55, TEXT, 1)
        put(img, b, (x - 60, 720), 0.5, MUTED, 1)
        if x < 1600:
            cv2.line(img, (x + 50, 590), (x + 320, 590), MUTED, 2)
        x += 420
    return img


def frame_answer():
    img = base("02  /  Grounded answer", "Qwen3 explains the result with citations")
    cv2.rectangle(img, (80, 215), (1840, 850), PANEL, -1)
    answer = ("The punishment for qatl-i-amd under section 302 is imprisonment for life or for a term "
              "which may extend to twenty-five years. The offence is not bailable or compoundable. "
              "Section 302 treats qatl-i-amd as the intentional killing of a person and sets the penalty "
              "within the criminal-law framework. [CrPC 1898.md#400]")
    wrapped(img, answer, (120, 285), 95, 0.72, TEXT, 36)
    put(img, "RETRIEVED SOURCES", (120, 735), 0.48, MUTED, 1)
    wrapped(img, "CrPC 1898.md#401   •   CrPC 1898.md#213   •   CrPC 1898.md#400", (120, 785), 105, 0.55, GREEN, 28)
    put(img, "Grounded answer  ✓", (1490, 920), 0.62, GREEN, 2)
    return img


def frame_phoenix():
    img = base("03  /  Observability", "Every step is visible in Arize Phoenix")
    cv2.rectangle(img, (80, 215), (780, 900), PANEL, -1)
    put(img, "TRACE TREE", (120, 265), 0.5, MUTED, 1)
    nodes = [("rag.query", 335, ORANGE), ("rag.route", 425, PURPLE), ("retrieval.hybrid", 515, GREEN), ("embedding.query", 605, PURPLE), ("retrieval.bm25", 695, GREEN), ("retrieval.hnsw", 785, GREEN), ("llm.generate_answer", 875, ORANGE)]
    for i, (name, y, c) in enumerate(nodes):
        if i:
            cv2.line(img, (145, y - 68), (165, y - 20), MUTED, 2)
        cv2.circle(img, (180, y), 12, c, -1)
        put(img, name, (215, y + 8), 0.62, TEXT, 1)
    cv2.rectangle(img, (850, 215), (1840, 900), PANEL, -1)
    put(img, "TRACE DETAILS", (890, 265), 0.5, MUTED, 1)
    metrics = [("Latency", "~12 s", ORANGE), ("Route", "SINGLE_HOP", PURPLE), ("Prompt tokens", "tracked", GREEN), ("Completion tokens", "tracked", GREEN), ("Sources", "6 chunks", GREEN), ("Phoenix project", "Local Hybrid RAG", ORANGE)]
    y = 355
    for label, value, c in metrics:
        cv2.rectangle(img, (890, y - 35), (1800, y + 30), (47, 47, 53), -1)
        put(img, label, (920, y + 5), 0.58, MUTED, 1)
        put(img, value, (1430, y + 5), 0.62, c, 2)
        y += 84
    return img


def frame_metrics():
    img = base("04  /  Evaluation", "Retrieval quality you can measure")
    cv2.rectangle(img, (80, 225), (1840, 875), PANEL, -1)
    put(img, "50-question retrieval benchmark", (120, 295), 0.75, TEXT, 2)
    put(img, "44 answerable questions  •  6 unanswerable questions excluded from retrieval scores", (120, 345), 0.55, MUTED, 1)
    vals = [("Recall", "0.95", "excellent coverage"), ("Precision", "0.59", "room to reduce extras"), ("NDCG", "0.96", "strong ranking"), ("MRR", "1.00", "correct result first"), ("Hit rate", "1.00", "all answerable queries hit")]
    y = 445
    for label, value, note in vals:
        put(img, label, (150, y), 0.68, TEXT, 1)
        cv2.rectangle(img, (440, y - 28), (1320, y + 10), (54, 54, 60), -1)
        width = int(880 * float(value))
        cv2.rectangle(img, (440, y - 28), (440 + width, y + 10), GREEN, -1)
        put(img, value, (1370, y), 0.7, GREEN, 2)
        put(img, note, (1510, y), 0.5, MUTED, 1)
        y += 78
    put(img, "Built locally • Chroma persistent vector store • Qwen3 on Ollama", (120, 830), 0.62, ORANGE, 2)
    return img


def main():
    out = Path("outputs/linkedin_hybrid_rag_demo.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
    slides = [frame_query(), frame_answer(), frame_phoenix(), frame_metrics()]
    for slide in slides:
        for _ in range(24 * 4):
            writer.write(slide)
    writer.release()
    print(out.resolve())


if __name__ == "__main__":
    main()
