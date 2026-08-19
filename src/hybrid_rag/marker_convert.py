from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def marker_executable() -> str:
    """Find the Marker CLI inside the active virtual environment."""
    candidate = Path(sys.executable).with_name("marker_single.exe")
    return str(candidate) if candidate.exists() else (shutil.which("marker_single") or "marker_single")


def convert_pdfs(
    input_dir: Path,
    output_dir: Path,
    mode: str = "fast",
    disable_ocr: bool = True,
) -> int:
    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF files found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = marker_executable()
    for pdf in pdfs:
        command = [
            executable, str(pdf), "--mode", mode, "--disable_multiprocessing",
            "--output_format", "markdown", "--output_dir", str(output_dir),
        ]
        if disable_ocr:
            command.append("--disable_ocr")
        subprocess.run(command, check=True)
    return len(pdfs)

