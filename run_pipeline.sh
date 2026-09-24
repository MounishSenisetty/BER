#!/usr/bin/env bash
# End-to-end reproducible run.
#   ./run_pipeline.sh                 -> uses data/train + data/test (real competition data)
#   SYNTHETIC=1 ./run_pipeline.sh     -> generates a synthetic dataset first and scores the hold-out
set -euo pipefail
cd "$(dirname "$0")"

TRAIN_DIR=${TRAIN_DIR:-data/train}
TEST_DIR=${TEST_DIR:-data/test}
MODEL_DIR=${MODEL_DIR:-models}
OUT_DIR=${OUT_DIR:-output}

if [[ "${SYNTHETIC:-0}" == "1" ]]; then
  python scripts/make_synthetic_data.py --out data
  EXTRA=(--gt data/test_labels/test_ground_truth.tsv)
else
  EXTRA=()
fi

python -m src.train --data-dir "$TRAIN_DIR" --model-dir "$MODEL_DIR"
python -m src.inference --data-dir "$TEST_DIR" --model-dir "$MODEL_DIR" --out-dir "$OUT_DIR" "${EXTRA[@]}"
echo "Deliverables: $OUT_DIR/matching_results.tsv  $OUT_DIR/candidate_pairs.tsv"
