import io
import numpy as np
import soundfile as sf
import librosa


def load_and_preprocess(file_bytes: bytes, target_sr: int = 16000):
    # soundfile works best for WAV; librosa helps with resampling
    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=True)
    # data shape: (samples, channels)

    # stereo -> mono
    mono = data.mean(axis=1)

    # resample
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    duration = float(len(mono)) / sr
    return mono, sr, duration