import json
from collections import defaultdict
from pathlib import Path
import random

def generate_test_manifest(train_manifest_path, output_path, samples_per_lang=5):
    with open(train_manifest_path) as f:
        data = json.load(f)

    lang_buckets = defaultdict(list)
    for item in data:
        lang_buckets[item["lang"]].append(item)

    test_manifest = []
    for lang, items in lang_buckets.items():
        test_manifest += random.sample(items, min(samples_per_lang, len(items)))

    with open(output_path, "w") as f:
        json.dump(test_manifest, f, indent=2)

    print(f"✅ test_manifest.json created with {len(test_manifest)} samples.")

# Usage:
# generate_test_manifest("train_manifest.json", "test_manifest.json")