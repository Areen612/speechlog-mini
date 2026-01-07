from typing import List

from pydantic import BaseModel


class Segment(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    text_norm: str


class TranscriptionResponse(BaseModel):
    duration_sec: float
    segments: List[Segment]
    latency_ms: int


class HealthResponse(BaseModel):
    ok: bool
