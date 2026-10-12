#!/bin/bash
set -e

BIN="/home/alessio/Documents/projects/flatzinc_to_graph/ZincToWl/out/bin/ZincToWl"
PYTHON="/home/alessio/envs/ml/bin/python3"
DATASET="data/pb_native_dataset.csv"
CACHE_DIR="data/features_pb_native"
RES_DIR="data/as-score-results/pb_native"

mkdir -p "$CACHE_DIR" "$RES_DIR"

for feat in wlc-2 wlce-2; do
  for fold in 0 1 2 3 4; do
    res_file="${RES_DIR}/rnd-forest-${feat}-${fold}-42.json"
    if [ ! -f "$res_file" ]; then
      echo ">>> Running $feat fold $fold"
      rm -f /tmp/zinctowl_8.sock
      ZINC_TO_WL_BIN="$BIN" "$PYTHON" src/algorithm_selection_pb/algorithm_selection.py \
        -f "$feat" \
        -m rnd-forest \
        --cv-fold "$fold" \
        --max-cv 5 \
        --rnd-state 42 \
        --dataset "$DATASET" \
        --cache-dir "$CACHE_DIR" \
        --result "$res_file"
    else
      echo ">>> $res_file already exists, skipping."
    fi
  done
done

echo ">>> All native order-2 experiments finished successfully!"
