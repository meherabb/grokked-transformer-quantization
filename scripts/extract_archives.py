#!/usr/bin/env python3
"""Extract the archived research ZIP files into data/extracted/.

The script never modifies or deletes the original archives.
"""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    ROOT / 'data/raw_archives/e1_dataset.zip': ROOT / 'data/extracted/e1_dataset',
    ROOT / 'data/raw_archives/e5_dataset.zip': ROOT / 'data/extracted/e5_dataset',
    ROOT / 'data/raw_archives/E6E7_DATASET_SLUG.zip': ROOT / 'data/extracted/e6e7',
    ROOT / 'data/consolidated/grokked_transformer_quantization_experiments_v1.zip': ROOT / 'data/extracted/consolidated',
}

for archive, destination in ARCHIVES.items():
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
    print(f'Extracted {archive.relative_to(ROOT)} -> {destination.relative_to(ROOT)}')
