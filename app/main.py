import time
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.audio_io import load_and_preprocess
from app.vad import SileroVADService
from app.vas import merge_and_pad_segments
from app.asr import ASRService
from app.normalize_ar import normalize_ar
from app.schemas import HealthResponse, TranscriptionResponse

app = FastAPI()
ASR: ASRService | None = None
VAD: SileroVADService | None = None


@app.on_event("startup")
def startup():
    global ASR, VAD
    ASR = ASRService(model_size="base")
    # Silero VAD is lightweight; keep it loaded
    VAD = SileroVADService(sampling_rate=16000)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return {"ok": True}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    if ASR is None or VAD is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    t0 = time.time()
    audio, sr, duration = load_and_preprocess(await file.read(), target_sr=16000)

    # VAD (Silero) returns coarse speech timestamps
    vad_segments = VAD.speech_timestamps_ms(audio)

    # VAS post-processing: pad + merge
    audio_duration_ms = int(duration * 1000)
    segments_ms = merge_and_pad_segments(
        vad_segments,
        audio_duration_ms=audio_duration_ms,
        min_silence_ms=300,
        pad_ms=120,
    )

    out = []
    for seg in segments_ms:
        s_ms = seg["start_ms"]
        e_ms = seg["end_ms"]
        s = int(s_ms * sr / 1000)
        e = int(e_ms * sr / 1000)
        chunk = audio[s:e]
        text = ASR.transcribe_segment(chunk)
        out.append(
            {
                "start_ms": s_ms,
                "end_ms": e_ms,
                "text": text,
                "text_norm": normalize_ar(text),
            }
        )

    latency_ms = int((time.time() - t0) * 1000)
    return {"duration_sec": duration, "segments": out, "latency_ms": latency_ms}
