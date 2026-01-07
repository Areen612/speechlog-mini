from __future__ import annotations

import json
from pathlib import Path

import librosa
import soundfile as sf
from datasets import Audio, config as ds_config, load_dataset
from tqdm import tqdm

DATASET_ID = "MohamedRashad/common-voice-18-arabic"  # Arabic-only Common Voice 18 ([huggingface.co](https://huggingface.co/datasets/MohamedRashad/common-voice-18-arabic))


def find_commonvoice_clips_dir() -> Path | None:
    cache_root = Path(ds_config.HF_DATASETS_CACHE)
    extracted_root = cache_root / "downloads" / "extracted"
    if not extracted_root.exists():
        return None

    matches = sorted(extracted_root.rglob("cv-corpus-*/ar/clips"))
    if matches:
        return matches[0]
    return None


def resolve_audio_path(audio: dict, ex: dict, clips_dir: Path | None) -> Path:
    candidates: list[Path] = []

    audio_path = audio.get("path")
    if audio_path:
        candidates.append(Path(audio_path))

    path_field = ex.get("path")
    if path_field:
        candidates.append(Path(path_field))

    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if clips_dir:
            if "clips" in candidate.parts:
                maybe = clips_dir.parent / candidate
            else:
                maybe = clips_dir / candidate
            if maybe.exists():
                return maybe
            maybe = clips_dir / candidate.name
            if maybe.exists():
                return maybe

    raise FileNotFoundError(
        "Could not resolve audio file. "
        f"audio.path={audio_path!r} ex.path={path_field!r} clips_dir={str(clips_dir)!r}"
    )


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

    # Load a small slice for speed and avoid HF audio decoding.
    ds = load_dataset(DATASET_ID, split=f"{split}[:{n}]")
    ds = ds.cast_column("audio", Audio(decode=False))
    clips_dir_root = find_commonvoice_clips_dir()

    with manifest_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds, desc=f"Saving {split}[:{n}]")):
            audio = ex["audio"]
            audio_path = resolve_audio_path(audio, ex, clips_dir_root)
            # librosa handles mono conversion and resampling
            arr, sr = librosa.load(audio_path, sr=target_sr, mono=True)

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
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {n} clips to: {clips_dir}")
    print(f"✅ Wrote manifest to: {manifest_path}")


if __name__ == "__main__":
    # quick defaults for interview prep
    main(split="validation", n=50)
