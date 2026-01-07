import io
import tempfile

import librosa
import soundfile as sf


def load_and_preprocess(file_bytes: bytes, target_sr: int = 16000):
    # soundfile works best for WAV; librosa helps with resampling
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=True)
        # data shape: (samples, channels)
        mono = data.mean(axis=1)

        # resample
        if sr != target_sr:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
    except Exception:
        # Fallback for compressed formats (e.g., mp3) using ffmpeg/audioread.
        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            mono, sr = librosa.load(tmp.name, sr=target_sr, mono=True)

    duration = float(len(mono)) / sr
    return mono, sr, duration
