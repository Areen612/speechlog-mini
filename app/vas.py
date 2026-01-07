from typing import List, Dict


def merge_and_pad_segments(
    vad_segments: List[Dict[str, int]],
    audio_duration_ms: int,
    min_silence_ms: int = 300,
    pad_ms: int = 120,
) -> List[Dict[str, int]]:
    """Input: [{'start_ms':..,'end_ms':..}] from VAD. Output: merged/padded segments."""
    if not vad_segments:
        return []

    # sort
    segs = sorted(vad_segments, key=lambda x: x["start_ms"])

    # pad
    padded = []
    for s in segs:
        start = max(0, s["start_ms"] - pad_ms)
        end = min(audio_duration_ms, s["end_ms"] + pad_ms)
        if end > start:
            padded.append({"start_ms": start, "end_ms": end})

    # merge
    merged = [padded[0]]
    for cur in padded[1:]:
        prev = merged[-1]
        gap = cur["start_ms"] - prev["end_ms"]
        if gap <= min_silence_ms:
            prev["end_ms"] = max(prev["end_ms"], cur["end_ms"])
        else:
            merged.append(cur)

    return merged