from datasets import load_dataset
import json
from pathlib import Path

def extract_commonvoice(lang, output_dir):
    ds = load_dataset("mozilla-foundation/common_voice_13_0", lang, split="train")
    manifest = []
    for sample in ds:
        if sample["audio"] and sample["sentence"]:
            manifest.append({
                "audio": sample["audio"]["path"],
                "text": sample["sentence"],
                "phonemes": "",  # fill via phonemizer
                "lang": lang
            })
    with open(Path(output_dir) / f"manifest_{lang}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Saved: manifest_{lang}.json")

# extract_commonvoice("hi", "data/")