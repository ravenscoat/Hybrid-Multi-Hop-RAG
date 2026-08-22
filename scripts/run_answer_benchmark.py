from __future__ import annotations

import json
from pathlib import Path

from gradio_client import Client


def main() -> None:
    input_path = Path("data/eval/random_answer_20.jsonl")
    output_path = Path("outputs/random_answer_20_results.json")
    client = Client("http://127.0.0.1:7860")
    rows = []
    for index, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        item = json.loads(line)
        result = client.predict(item["question"], "Automatic", False, api_name="/ask")
        rows.append(
            {
                "index": index,
                "expected_route": item["expected_route"],
                "question": item["question"],
                "answer": result[0],
                "actual_route": result[1],
                "sources": result[2],
            }
        )
        print(f"{index}/20 {result[1]} {result[0][:180].replace(chr(10), ' ')}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    correct = sum(row["expected_route"] == row["actual_route"] for row in rows)
    print(f"route_accuracy={correct}/{len(rows)}")
    print(output_path)


if __name__ == "__main__":
    main()
