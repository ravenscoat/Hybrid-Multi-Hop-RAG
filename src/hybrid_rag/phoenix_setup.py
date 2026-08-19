from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_tracer():
    os.environ.setdefault("PHOENIX_WORKING_DIR", str(Path(".phoenix").resolve()))
    from phoenix.otel import register

    project = os.getenv("PHOENIX_PROJECT_NAME", "Local Hybrid RAG")
    register(project_name=project, auto_instrument=False)
    from opentelemetry import trace

    return trace.get_tracer("local-hybrid-rag")
