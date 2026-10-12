#!/bin/bash
set -e

BIN="/home/alessio/Documents/projects/flatzinc_to_graph/ZincToWl/out/bin/ZincToWl"
PYTHON="/home/alessio/envs/ml/bin/python3"
DATASET="data/pb_dataset.csv"
CACHE_DIR="data/features_pb"
RES_DIR="data/as-score-results/pb"
SEED=42

mkdir -p "$CACHE_DIR" "$RES_DIR"

# # Run Random Forest across all standard cut and base WL representations
# for feat in wlc-1 wlce-1 wlc-2 wlce-2; do
#   for fold in 0 1 2 3 4; do
#     res_file="${RES_DIR}/rnd-forest-${feat}-${fold}-${SEED}.json"
#     if [ ! -f "$res_file" ]; then
#       echo ">>> Running PB: model=rnd-forest feat=$feat fold=$fold"
#       rm -f /tmp/zinctowl_8.sock
#       ZINC_TO_WL_BIN="$BIN" "$PYTHON" src/algorithm_selection_pb/algorithm_selection.py \
#         -f "$feat" \
#         -m rnd-forest \
#         --cv-fold "$fold" \
#         --max-cv 5 \
#         --rnd-state "$SEED" \
#         --dataset "$DATASET" \
#         --cache-dir "$CACHE_DIR" \
#         --result "$res_file"
#     else
#       echo ">>> $res_file already exists, skipping."
#     fi
#   done
# done

# # Run SVC and NN across cut representations
# for model in svc nn; do
#   for feat in wlc-1 wlce-1 wlc-2 wlce-2; do
#     for fold in 0 1 2 3 4; do
#       res_file="${RES_DIR}/${model}-${feat}-${fold}-${SEED}.json"
#       if [ ! -f "$res_file" ]; then
#         echo ">>> Running PB: model=$model feat=$feat fold=$fold"
#         rm -f /tmp/zinctowl_8.sock
#         ZINC_TO_WL_BIN="$BIN" "$PYTHON" src/algorithm_selection_pb/algorithm_selection.py \
#           -f "$feat" \
#           -m "$model" \
#           --cv-fold "$fold" \
#           --max-cv 5 \
#           --rnd-state "$SEED" \
#           --dataset "$DATASET" \
#           --cache-dir "$CACHE_DIR" \
#           --result "$res_file"
#       else
#         echo ">>> $res_file already exists, skipping."
#       fi
#     done
#   done
# done

# # Run fzn2feat baseline across all models
# for model in rnd-forest svc nn; do
#   for fold in 0 1 2 3 4; do
#     res_file="${RES_DIR}/${model}-fzn2feat-${fold}-${SEED}.json"
#     if [ ! -f "$res_file" ]; then
#       echo ">>> Running PB: model=$model feat=fzn2feat fold=$fold"
#       rm -f /tmp/zinctowl_8.sock
#       ZINC_TO_WL_BIN="$BIN" "$PYTHON" src/algorithm_selection_pb/algorithm_selection.py \
#         -f fzn2feat \
#         -m "$model" \
#         --cv-fold "$fold" \
#         --max-cv 5 \
#         --rnd-state "$SEED" \
#         --dataset "$DATASET" \
#         --cache-dir "$CACHE_DIR" \
#         --result "$res_file"
#     else
#       echo ">>> $res_file already exists, skipping."
#     fi
#   done
# done

# Run encoder baseline across all models
for model in rnd-forest svc nn; do
  for fold in 0 1 2 3 4; do
    res_file="${RES_DIR}/${model}-encoder-${fold}-${SEED}.json"
    if [ ! -f "$res_file" ]; then
      echo ">>> Running PB: model=$model feat=encoder fold=$fold"
      rm -f /tmp/zinctowl_8.sock
      ZINC_TO_WL_BIN="$BIN" "$PYTHON" src/algorithm_selection_pb/algorithm_selection.py \
        -f encoder \
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

echo ">>> All PB algorithm selection experiments finished successfully!"
