#!/usr/bin/env python3
"""Lightweight structural validation for the research repository."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
required = [
    'README.md', 'CITATION.cff', 'LICENSE', 'LICENSE-DATA.md',
    'notebooks/e1_addition_experiments.ipynb',
    'notebooks/e5_multiplication_subtraction_experiments.ipynb',
    'notebooks/grokked_transformer_quantization_analysis_v1.ipynb',
    'data/raw_archives/e1_dataset.zip',
    'data/raw_archives/e5_dataset.zip',
    'data/raw_archives/E6E7_DATASET_SLUG.zip',
    'data/consolidated/grokked_transformer_quantization_experiments_v1.zip',
    'figures/main/pdf/fig1_grokking_trajectory.pdf',
    'figures/main/pdf/fig2_primary_selectivity.pdf',
    'figures/main/pdf/fig3_architectural_fragility.pdf',
    'figures/main/pdf/fig4_cross_task_robustness.pdf',
    'figures/supplementary/pdf/figS1_phenomenon.pdf',
    'figures/supplementary/pdf/figS7_theory_verification.pdf',
]
errors = []
for rel in required:
    if not (ROOT / rel).exists():
        errors.append(f'Missing required file: {rel}')

# Ensure notebooks are valid JSON.
for nb in (ROOT / 'notebooks').glob('*.ipynb'):
    try:
        with nb.open(encoding='utf-8') as fh:
            obj = json.load(fh)
        if 'cells' not in obj:
            errors.append(f'Notebook has no cells array: {nb.relative_to(ROOT)}')
    except Exception as exc:
        errors.append(f'Invalid notebook JSON {nb.relative_to(ROOT)}: {exc}')

# Catch accidental duplicate extensions and GitHub hard file-size limit.
for path in ROOT.rglob('*'):
    if not path.is_file():
        continue
    lower = path.name.lower()
    if lower.endswith('.zip.zip') or lower.endswith('.ipynb.ipynb'):
        errors.append(f'Duplicate extension: {path.relative_to(ROOT)}')
    if path.stat().st_size > 100 * 1024 * 1024:
        errors.append(f'File exceeds GitHub 100 MB limit: {path.relative_to(ROOT)}')

if errors:
    print('\n'.join(errors))
    sys.exit(1)
print('Repository structure validated successfully.')
