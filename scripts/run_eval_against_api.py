from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from tqdm import tqdm


def main(
    manifest_path: str = "data/commonvoice_ar/eval/manifest/commonvoice_ar_validation_50.jsonl",
    api_url: str = "http://127.0.0.1:8000/transcribe",
    out_dir: str = "runs/commonvoice_eval",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out / f"predictions_{ts}.jsonl"

    manifest = Path(manifest_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    with out_path.open("w", encoding="utf-8") as f:
        for r in tqdm(rows, desc="Calling /transcribe"):
            wav_path = Path(r["path"])
            with wav_path.open("rb") as audio:
                files = {"file": (wav_path.name, audio, "audio/wav")}
                resp = requests.post(api_url, files=files, timeout=120)
            resp.raise_for_status()

            pred = resp.json()
            # store minimal useful fields for analysis
            rec = {
                "id": r["id"],
                "path": r["path"],
                "ref_text": r.get("ref_text", ""),
                "pred": pred,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "")

    print(f"✅ Wrote predictions to: {out_path}")


if __name__ == "__main__":
    main()