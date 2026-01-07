from __future__ import annotations

import json
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "MohamedRashad/common-voice-18-arabic"  # Arabic-only Common Voice 18 ([huggingface.co](https://huggingface.co/datasets/MohamedRashad/common-voice-18-arabic))


def to_mono(x):
    # HF audio arrays can be (n,) or (n, channels)
    if hasattr(x, "ndim") and x.ndim == 2:
        return x.mean(axis=1)
    return x


def main(
    split: str = "validation",
    n: int = 50,
    out_dir: str = "data/commonvoice_ar/eval",
    target_sr: int = 16000,
):
    out = Path(out_dir)
    clips_dir = out / "clips"
    manifest_dir = out / "manifest"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f"commonvoice_ar_{split}_{n}.jsonl"

    # Load a small slice for speed 
    ds = load_dataset(DATASET_ID, split=f"{split}[:{n}]")

    with manifest_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds, desc=f"Saving {split}[:{n}]")):
            audio = ex["audio"]
            arr = to_mono(audio["array"])
            sr = audio["sampling_rate"]

            # Resample to 16k for your pipeline
            if sr != target_sr:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            clip_name = f"cv18_ar_{split}_{i:05d}.wav"
            clip_path = clips_dir / clip_name
            sf.write(clip_path, arr, sr)

            record = {
                "id": f"{split}_{i:05d}",
                # store relative path (repo-friendly)
                "path": str(clip_path.as_posix()),
                "ref_text": ex.get("sentence", ""),
                "split": split,
                "source": DATASET_ID,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "")

    print(f"✅ Wrote {n} clips to: {clips_dir}")
    print(f"✅ Wrote manifest to: {manifest_path}")


if __name__ == "__main__":
    # quick defaults for interview prep
    main(split="validation", n=50)