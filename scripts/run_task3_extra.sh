#!/usr/bin/env bash
# Start one or more EXTRA Task 3 runs in the background (they keep running if the browser closes).
#   bash scripts/run_task3_extra.sh long100 long100_diffaug
# Each RUN reads task3_gan/$MEMBER/config_$RUN.yaml and writes ONLY to task3_gan/$MEMBER/runs/$RUN/
# (executed notebook, checkpoints, outputs, submission.csv). The main run is never touched.
# Re-running the same command after an interruption RESUMES from runs/$RUN/checkpoints/last.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
MEMBER="${MEMBER:-ruchika_chaurasia}"
NB="task3_gan/$MEMBER/src/task3_gan_v2.ipynb"
[ -f "$NB" ] || NB="task3_gan/$MEMBER/src/task3_gan.ipynb"
for RUN in "$@"; do
  [ -f "task3_gan/$MEMBER/config_$RUN.yaml" ] || { echo "missing task3_gan/$MEMBER/config_$RUN.yaml"; exit 1; }
  OUT="task3_gan/$MEMBER/runs/$RUN"
  mkdir -p "$OUT"
  nohup env RUN="$RUN" jupyter nbconvert --to notebook --execute "$NB" \
      --output-dir "$OUT" --output "task3_gan_$RUN.ipynb" --ExecutePreprocessor.timeout=-1 \
      > "$OUT/console.log" 2>&1 &
  echo "started $RUN (pid $!)  ->  progress: tail -f reproducibility/raw_logs/\$(ls -t reproducibility/raw_logs | grep ${MEMBER}_$RUN | head -1)"
  sleep 3
done
