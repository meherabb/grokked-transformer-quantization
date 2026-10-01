#!/usr/bin/env python3
"""Regenerate manifests/checksums_sha256.txt for repository artifacts."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manifests/checksums_sha256.txt'
EXCLUDE = {OUT.resolve()}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

rows = []
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or path.resolve() in EXCLUDE or '.git' in path.parts:
        continue
    rows.append(f'{digest(path)}  {path.relative_to(ROOT).as_posix()}')

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text('\n'.join(rows) + '\n', encoding='utf-8')
print(f'Wrote {len(rows)} checksums to {OUT.relative_to(ROOT)}')
