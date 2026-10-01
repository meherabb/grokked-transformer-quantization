#!/usr/bin/env python3
"""Verify manifests/checksums_sha256.txt."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'manifests/checksums_sha256.txt'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

failures = []
for line in MANIFEST.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    expected, rel = line.split('  ', 1)
    path = ROOT / rel
    if not path.exists():
        failures.append(f'MISSING: {rel}')
        continue
    actual = digest(path)
    if actual != expected:
        failures.append(f'MISMATCH: {rel}')

if failures:
    print('\n'.join(failures))
    sys.exit(1)
print('All checksums verified.')
