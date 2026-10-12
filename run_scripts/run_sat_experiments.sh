#!/bin/bash
set -e

BIN="/home/alessio/Documents/projects/flatzinc_to_graph/ZincToWl/out/bin/ZincToWl"
PYTHON="/home/alessio/envs/ml/bin/python3"
DATASET="data/sat/sat_dataset.csv"
CACHE_DIR="data/features_sat"
RES_DIR="data/as-score-results/sat"
SEED=42

mkdir -p "$CACHE_DIR" "$RES_DIR"

# Run Random Forest across all standard cut and base WL representations
for feat in wlc-1 wlce-1 wlc-2 wlce-2; do
  for fold in 0 1 2 3 4; do
    res_file="${RES_DIR}/rnd-forest-${feat}-${fold}-${SEED}.json"
    if [ ! -f "$res_file" ]; then
      echo ">>> Running SAT: model=rnd-forest feat=$feat fold=$fold"
      rm -f /tmp/zinctowl_8.sock
      ZINC_TO_WL_BIN="$BIN" "$PYTHON" src/algorithm_selection_sat/algorithm_selection.py \
        -f "$feat" \
        -m rnd-forest \
        --cv-fold "$fold" \
        --max-cv 5 \
        --rnd-state "$SEED" \
        --dataset "$DATASET" \
        --cache-dir "$CACHE_DIR" \
        --result "$res_file"
    else
      echo ">>> $res_file already exists, skipping."
    fi
  done
done

# Run SVC and NN across cut representations
for model in svc nn; do
  for feat in wlc-1 wlce-1 wlc-2 wlce-2; do
    for fold in 0 1 2 3 4; do
      res_file="${RES_DIR}/${model}-${feat}-${fold}-${SEED}.json"
      if [ ! -f "$res_file" ]; then
        echo ">>> Running SAT: model=$model feat=$feat fold=$fold"
        rm -f /tmp/zinctowl_8.sock
        ZINC_TO_WL_BIN="$BIN" "$PYTHON" src/algorithm_selection_sat/algorithm_selection.py \
          -f "$feat" \
          -m "$model" \
          --cv-fold "$fold" \
          --max-cv 5 \
          --rnd-state "$SEED" \
          --dataset "$DATASET" \
          --cache-dir "$CACHE_DIR" \
          --result "$res_file"
      else
        echo ">>> $res_file already exists, skipping."
      fi
    done
  done
done

echo ">>> All SAT algorithm selection experiments finished successfully!"

# Run fzn2feat baseline across all models
echo ">>> Pre-extracting fzn2feat for SAT instances..."
"$PYTHON" extract_sat_fzn2feat.py

for model in rnd-forest svc nn; do
  for fold in 0 1 2 3 4; do
    res_file="${RES_DIR}/${model}-fzn2feat-${fold}-${SEED}.json"
    if [ ! -f "$res_file" ]; then
      echo ">>> Running SAT: model=$model feat=fzn2feat fold=$fold"
      rm -f /tmp/zinctowl_8.sock
      ZINC_TO_WL_BIN="$BIN" "$PYTHON" src/algorithm_selection_sat/algorithm_selection.py \
        -f fzn2feat \
        -m "$model" \
        --cv-fold "$fold" \
        --max-cv 5 \
        --rnd-state "$SEED" \
        --dataset "$DATASET" \
        --cache-dir "$CACHE_DIR" \
        --result "$res_file"
    else
      echo ">>> $res_file already exists, skipping."
    fi
  done
done

echo ">>> All experiments including fzn2feat completed successfully!"
