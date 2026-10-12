#!/bin/bash
set -e

BIN="/home/alessio/Documents/projects/flatzinc_to_graph/ZincToWl/out/bin/ZincToWl"
PYTHON="/home/alessio/envs/ml/bin/python3"
DATASET="data/data_sequential_5solvers.csv"
CACHE_DIR="data/mzn_5solvers_features_cache"
RES_DIR="data/as-score-results/mzn_challenge"
SEED=42
YEARS=(2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026)

mkdir -p "$CACHE_DIR" "$RES_DIR"

# Allow selecting a single year or defaulting to all 15 challenge years
if [ -n "$1" ]; then
  YEARS=("$1")
fi

echo ">>> Running MiniZinc Challenge Algorithm Selection for years: ${YEARS[*]}"

# Run Random Forest across representations
for year in "${YEARS[@]}"; do
  for feat in wlc-1 wlce-1 wlc-2 wlce-2 fzn2feat; do
    for fold in 0 1 2 3 4; do
      res_file="${RES_DIR}/rnd-forest-${feat}-${year}-${fold}-${SEED}.json"
      if [ ! -f "$res_file" ]; then
        echo ">>> Running MZN: year=$year model=rnd-forest feat=$feat fold=$fold"
        rm -f /tmp/zinctowl_mzn.sock
        ZINC_TO_WL_BIN="$BIN" "$PYTHON" src/algorithm_selection_mzn_challenge/algorithm_selection.py \
          -f "$feat" \
          -m rnd-forest \
          --year "$year" \
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

# Run SVC and NN models
for model in svc nn; do
  for year in "${YEARS[@]}"; do
    for feat in wlc-1 wlce-1 wlc-2 wlce-2 fzn2feat; do
      for fold in 0 1 2 3 4; do
        res_file="${RES_DIR}/${model}-${feat}-${year}-${fold}-${SEED}.json"
        if [ ! -f "$res_file" ]; then
          echo ">>> Running MZN: year=$year model=$model feat=$feat fold=$fold"
          rm -f /tmp/zinctowl_mzn.sock
          ZINC_TO_WL_BIN="$BIN" "$PYTHON" src/algorithm_selection_mzn_challenge/algorithm_selection.py \
            -f "$feat" \
            -m "$model" \
            --year "$year" \
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
done

echo ">>> All MiniZinc Challenge algorithm selection experiments finished successfully!"
