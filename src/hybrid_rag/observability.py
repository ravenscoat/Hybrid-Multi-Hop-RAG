from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


def _attribute_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, int, float, bool)) for item in value
    ):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


@dataclass
class TraceSpan:
    span: Any = None

    @property
    def span_id(self) -> str | None:
        if self.span is None:
            return None
        context = self.span.get_span_context()
        return format(context.span_id, "016x") if context.is_valid else None

    def set(self, key: str, value: Any) -> None:
        if self.span is not None and value is not None:
            self.span.set_attribute(key, _attribute_value(value))

    def output(self, value: Any) -> None:
        self.set("output.value", value)
        self.set("output.mime_type", "application/json" if not isinstance(value, str) else "text/plain")

    def event(self, name: str, attributes: dict | None = None) -> None:
        if self.span is not None:
            self.span.add_event(name, {key: _attribute_value(value) for key, value in (attributes or {}).items()})


@contextmanager
def rag_span(
    name: str, attributes: dict | None = None, span_kind: str = "CHAIN"
) -> Iterator[TraceSpan]:
    """Create a Phoenix-compatible span when Phoenix is enabled.

    The RAG still works normally when the optional Phoenix packages are absent.
    Set PHOENIX_ENABLED=true to export spans to a local or cloud Phoenix server.
    """
    if os.getenv("PHOENIX_ENABLED", "false").casefold() != "true":
        yield TraceSpan()
        return
    try:
        from .phoenix_setup import get_tracer
        tracer = get_tracer()
    except ImportError:
        # Observability must never make the local RAG unavailable.
        yield TraceSpan()
        return
    from opentelemetry.trace import Status, StatusCode

    with tracer.start_as_current_span(name) as span:
        span.set_attribute("openinference.span.kind", span_kind)
        for key, value in (attributes or {}).items():
            span.set_attribute(key, _attribute_value(value))
        yield TraceSpan(span)
        span.set_status(Status(StatusCode.OK))
