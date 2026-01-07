from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from silero_vad import load_silero_vad, get_speech_timestamps


@dataclass
class SileroVADService:
    sampling_rate: int = 16000

    def __post_init__(self):
        # Loads a lightweight VAD model
        self.model = load_silero_vad()

    def speech_timestamps_ms(self, audio_mono: np.ndarray) -> List[Dict[str, int]]:
        """Return [{'start_ms': int, 'end_ms': int}, ...]"""
        wav = audio_mono.astype(np.float32)

        # return_seconds=True means start/end are in seconds (floats)
        raw = get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=self.sampling_rate,
            return_seconds=True,
        )

        out = []
        for seg in raw:
            # seg usually has keys: 'start', 'end'
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            if end_ms > start_ms:
                out.append({"start_ms": start_ms, "end_ms": end_ms})
        return out