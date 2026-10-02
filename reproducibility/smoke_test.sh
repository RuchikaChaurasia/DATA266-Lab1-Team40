#!/usr/bin/env bash
# ONE-COMMAND SMOKE TEST (run from the repo root):   bash reproducibility/smoke_test.sh
# Executes the Task 1 notebook of lineup A end to end on a tiny setting (1,500 stories from the
# TinyStories validation split, 2-layer model, 2 epochs) in a few minutes on CPU or GPU.
# Real results are untouched: everything goes to _smoke/ and to *_smoke_* logs/manifests.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -q -r requirements.txt
mkdir -p _smoke
SMOKE=1 jupyter nbconvert --to notebook --execute task1_llm/ruchika_chaurasia/src/task1_llm.ipynb \
  --output-dir _smoke --output task1_smoke_executed.ipynb --ExecutePreprocessor.timeout=1800
echo "Smoke test passed. Outputs: _smoke/task1_llm/ruchika_chaurasia/  | executed notebook: _smoke/task1_smoke_executed.ipynb"
