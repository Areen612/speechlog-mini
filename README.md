# Speechlog Mini

Speechlog Mini is a compact, end-to-end speech analytics pipeline that mirrors real CX speech products:

Audio -> preprocess -> VAD -> VAS -> ASR -> Arabic normalization -> JSON API -> (optional) Docker

It also includes scripts to build a small Common Voice Arabic evaluation pack and run your API against it.

---

## What this project demonstrates
- Systems-level thinking across a full audio pipeline
- Real-world audio handling (format, sample rate, segmentation)
- Integration of modern components (PyTorch VAD + Whisper ASR)
- A working FastAPI service with reproducible evaluation runs

---

## Tech stack
- FastAPI + Uvicorn: serve the pipeline as an API (`/transcribe`)
- Silero VAD (PyTorch): speech region detection (timestamps)
- VAS post-processing: merge + pad segments for stable ASR
- faster-whisper: optimized ASR (speech -> text) for CPU inference
- librosa + soundfile + numpy: audio loading/resampling utilities
- Poetry: dependency management and reproducible environments

---

## Project structure
```text
speechlog-mini/
  app/
    main.py              # FastAPI service
    audio_io.py          # audio loading, mono conversion, resampling to 16k
    vad.py               # Silero VAD wrapper -> speech timestamps (ms)
    vas.py               # merge + pad VAD segments -> final utterances
    asr.py               # faster-whisper ASR service
    normalize_ar.py      # minimal Arabic normalization
    schemas.py           # optional pydantic schemas for JSON responses
  tests/
    test_vas.py          # VAS unit tests
    test_metrics.py      # metrics tests (optional)
  scripts/
    prepare_commonvoice_eval.py  # build eval pack (mono 16k wav + manifest)
    run_eval_against_api.py      # call /transcribe and save outputs
  data/
    commonvoice_ar/
      eval/
        clips/           # generated wav files (mono 16k)
        manifest/        # JSONL manifest with ref_text
  runs/                  # generated predictions (JSONL)
  pyproject.toml
  poetry.lock
  Dockerfile             # optional
```

> **⚠️ Important:** `data/` and `runs/` are generated locally and should not be committed to version control.

---

## Requirements
- Python 3.11+
- Poetry
- ffmpeg (recommended for broader audio format support)

Install tools (macOS):
```bash
brew install ffmpeg poetry
```

---

## Setup
From repo root:

```bash
poetry install
poetry shell
```

If you prefer not to enter a shell, prefix commands with `poetry run`.

---

## Run the API
Start the server:

```bash
poetry run uvicorn app.main:app --reload
```

Health check:

- `GET http://127.0.0.1:8000/health`

---

## API usage: `/transcribe`
Example request:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@path/to/audio.wav"
```

Example response shape:

```json
{
  "duration_sec": 12.34,
  "segments": [
    {
      "start_ms": 900,
      "end_ms": 4200,
      "text": "....",
      "text_norm": "...."
    }
  ],
  "latency_ms": 1240
}
```

---

## Common Voice Arabic evaluation
Dataset: `MohamedRashad/common-voice-18-arabic`

### 1) Add deps (if not already in `pyproject.toml`)
```bash
poetry add datasets tqdm
```

### 2) Generate eval clips + manifest
Downloads a small slice (default: 50 clips), converts to mono 16k WAV, and writes a JSONL manifest.

```bash
poetry run python scripts/prepare_commonvoice_eval.py
```

Expected outputs:
- `data/commonvoice_ar/eval/clips/*.wav`
- `data/commonvoice_ar/eval/manifest/commonvoice_ar_validation_50.jsonl`

### 3) Run API evaluation
Start the API:

```bash
poetry run uvicorn app.main:app --reload
```

Run the evaluator:

```bash
poetry run python scripts/run_eval_against_api.py
```

Outputs:
- `runs/commonvoice_eval/predictions_YYYYMMDD_HHMMSS.jsonl`

---

## Troubleshooting
If transcripts look wrong:
- Verify audio is mono and 16kHz
- Check VAD aggressiveness and VAS padding
- Reduce fragmentation by adjusting segmentation thresholds
- Expect a domain shift on phone/compressed audio

If latency is high:
- Ensure models load once at startup (not per request)
- Reduce segment count (merge more or tune segmentation)
- Try a smaller Whisper model (`tiny` or `base`)

---

## Docker (optional)
Build:

```bash
docker build -t speechlog-mini .
```

Run:

```bash
docker run -p 8000:8000 speechlog-mini
```

---

## Notes
- This project is intended for learning and experimentation only, not for large-scale training or deployment in production environments. 
- It may lack the robustness, scalability, and security features required for production use, and users should exercise caution when applying it beyond educational purposes.
- Common Voice is used only to create a small, reproducible evaluation pack.
