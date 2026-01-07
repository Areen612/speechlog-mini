from app.vas import merge_and_pad_segments


def test_merge_and_pad_segments_merges_close_gaps():
    vad_segments = [
        {"start_ms": 1000, "end_ms": 1500},
        {"start_ms": 1600, "end_ms": 2100},  # gap=100ms (should merge if min_silence>=100)
        {"start_ms": 4000, "end_ms": 4500},
    ]
    out = merge_and_pad_segments(vad_segments, audio_duration_ms=5000, min_silence_ms=200, pad_ms=0)
    assert len(out) == 2
    assert out[0]["start_ms"] == 1000 and out[0]["end_ms"] == 2100


def test_merge_and_pad_segments_padding_clamps_to_bounds():
    vad_segments = [{"start_ms": 50, "end_ms": 100}]
    out = merge_and_pad_segments(vad_segments, audio_duration_ms=120, min_silence_ms=0, pad_ms=200)
    assert out[0]["start_ms"] == 0
    assert out[0]["end_ms"] == 120