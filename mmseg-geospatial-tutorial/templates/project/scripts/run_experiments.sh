#!/usr/bin/env bash
# Train several model configs of one project in sequence and evaluate each on the test split.
# Usage: bash scripts/run_experiments.sh <project_dir> [config names...]   (run from the tutorial root)
#   bash projects/buildings/scripts/run_experiments.sh projects/buildings deeplabv3plus_r50 segformer_b2
set -euo pipefail
PROJECT="${1:?project directory}"; shift
NAME="$(basename "$PROJECT")"
CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=($(ls "$PROJECT/configs"/*.py | xargs -n1 basename | sed 's/\.py$//'))
for c in "${CONFIGS[@]}"; do
  WD="work_dirs/$NAME/$c"
  echo "=============== $NAME / $c -> $WD"
  python tools/train.py "$PROJECT/configs/$c.py" --work-dir "$WD" 2>&1 | tee "$WD.train.log" || { echo "FAILED: $c"; continue; }
  python tools/test.py "$PROJECT/configs/$c.py" "$WD" --work-dir "$WD/test" 2>&1 | tee "$WD.test.log"
done
python tools/compare_runs.py "work_dirs/$NAME" --out "work_dirs/$NAME/summary.md"
