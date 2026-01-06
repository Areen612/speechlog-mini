#!/usr/bin/env bash
set -euo pipefail

# Run this from the parent directory where you want the project folder created.

# 1) Install system tools
brew install ffmpeg poetry

# 2) Create project folder
mkdir speechlog-mini && cd speechlog-mini

# 3) Initialize Poetry project (creates pyproject.toml)
poetry init -n

# 4) Add runtime dependencies
# fastapi         -> web framework (build the API)
# uvicorn         -> ASGI server that runs the FastAPI app
# python-multipart-> enables file upload (UploadFile) in FastAPI
# pydantic        -> request/response validation + schemas
poetry add fastapi uvicorn python-multipart pydantic

# numpy           -> numeric arrays (audio waveform as np.ndarray)
# soundfile       -> read/write WAV (fast + reliable for WAV)
# librosa         -> resampling + audio utilities (e.g., convert sr→16k)
poetry add numpy soundfile librosa

# torch           -> deep learning framework (needed by Silero VAD)
# torchaudio      -> audio helpers used in the PyTorch ecosystem
# silero-vad      -> Voice Activity Detection model (outputs speech timestamps)
poetry add torch torchaudio silero-vad

# faster-whisper  -> faster inference Whisper ASR (CTranslate2 backend);
#                  used to transcribe speech->text efficiently on CPU
poetry add faster-whisper

# datasets        -> Hugging Face datasets library; includes utilities for downloading datasets
#                  and loading audio files from paths
poetry add datasets 


# 5) Dev dependencies (tests)
poetry add --group dev pytest

# 6) Enter Poetry venv (choose one)
poetry shell
# OR run commands without entering shell:
# poetry run uvicorn app.main:app --reload

