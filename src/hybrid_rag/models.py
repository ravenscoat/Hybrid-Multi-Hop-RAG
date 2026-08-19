from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    id: int
    source: str
    text: str
    ordinal: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Chunk":
        return cls(**value)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    matched_query: str | None = None

