import argparse
from typing import Literal
import pandas as pd
import sys
import os
import json
import time

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train_torch_neural_network import train_and_test_nn_torch
from sklearn.model_selection import StratifiedKFold
from train_as_forest import train_and_test_rnd_forest
from train_as_svc import train_and_test_svc
from copy import deepcopy
import numpy as np
from feature_extraction import get_features
import subprocess

import warnings
warnings.filterwarnings("ignore")

import random
random.seed(42)
np.random.seed(42)

DEFAULT_DATASET = './data/car_sequencing_dataset.csv'

def load_data(data_path: str = DEFAULT_DATASET) -> pd.DataFrame:
    """
    Loads the Car Sequencing algorithm selection dataset.
    """
    data = pd.read_csv(data_path)
    return data

def data_to_list(data: pd.DataFrame) -> list[dict]:
    dict_data = []
    meta_cols = {'flatzinc', 'best_solver', 'min_time', 'max_time', 'problem', 'year', 'instance', 'name', 'gap', 'gap_strata'}
    solvers = [c for c in data.columns if c not in meta_cols]
    solver_to_idx = {s: i for i, s in enumerate(solvers)}

    for i in range(len(data)):
        d = data.iloc[i]
        fzn = str(d['flatzinc'])
        best_solver = str(d['best_solver'])
        label = solver_to_idx[best_solver]

        inst = {
            'flatzinc': fzn,
            'graph': fzn,
            'model': 'car_sequencing',
            'name': os.path.basename(fzn),
            'label': label,
            'best_solver': best_solver,
            'min_time': float(d['min_time']),
            'max_time': float(d['max_time']),
            'solvers': solvers,
        }
        for s in solvers:
            inst[s] = float(d[s])
        dict_data.append(inst)

    return dict_data

def split_stratified_by_gap(df: pd.DataFrame, rnd_state: int, current_fold: int, max_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert max_fold > 0, f'max fold must be > 0. got {max_fold}'
    assert current_fold >= 0, f'current fold must positive. got {current_fold}'
    assert current_fold < max_fold, f'current fold must be < max fold. got current fold:{current_fold} and max fold: {max_fold}'

    df_copy = df.copy()
    df_copy['gap'] = df_copy['max_time'] - df_copy['min_time']
    df_copy['gap_strata'] = pd.qcut(df_copy['gap'], q=5, labels=False, duplicates='drop')

    skf = StratifiedKFold(n_splits=max_fold, shuffle=True, random_state=rnd_state)
    idxs = [(train_idx, test_idx) for (train_idx, test_idx) in skf.split(df_copy, df_copy['gap_strata'])]
    train_idx, test_idx = idxs[current_fold]
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    return train_df, test_df

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--features', type=str, required=True, choices=['wlce-1', 'wlce-2', 'wlc-1', 'wlc-2', 'wlcu-1', 'wlcu-2', 'wlceu-1', 'wlceu-2', 'wl-1', 'wl-2', 'wln-1', 'wln-2', 'wlun-1', 'wlun-2', 'wle-0', 'wle-1', 'wle-2', 'wlne-1', 'wlne-2', 'wlune-1', 'wlune-2', 'sat-features', 'fzn2feat', 'combined'])
parser.add_argument('-m', '--model', type=str, required=True, choices=['svc', 'rnd-forest', 'nn', 'gb'])
parser.add_argument('--cv-fold', required=True, type=int)
parser.add_argument('--max-cv', required=True, type=int)
parser.add_argument('--result', required=True, type=str)
parser.add_argument('--rnd-state', required=True, type=int)
parser.add_argument('--pca', required=False, action='store_true', help='Apply PCA to reduce the number of features')
parser.add_argument('--all-levels', action='store_true', help='Use a concatenation of all aggregation levels instead of only the last one')
parser.add_argument('--dataset', type=str, default='./data/car_sequencing_dataset.csv', help='Path to the Car Sequencing dataset CSV')

def main():
    args = parser.parse_args()
    features_type = args.features
    model = args.model
    fold = args.cv_fold
    output_file = args.result
    max_cv = args.max_cv
    rnd_state = args.rnd_state
    all_levels = args.all_levels

    data = load_data(args.dataset)
    print(f'Loaded Car Sequencing dataset with {len(data)} instances, starting to compute features')

    train_df, test_df = split_stratified_by_gap(
        df=data,
        rnd_state=rnd_state,
        current_fold=fold,
        max_fold=max_cv
    )

    train_data = data_to_list(train_df)
    test_data = data_to_list(test_df)

    feat_cache_file = f'data/features_car_sequencing/{features_type.replace("-","")}-{fold}-{rnd_state}.csv'
    train_data, test_data = get_features(train_data, test_data, features_type, all_levels, feat_cache_file)
    print(f'Computed {features_type} features ({len(train_data[0]["features"])} dims), starting to train {model}')

    if model == 'rnd-forest':
        res = train_and_test_rnd_forest(train_data, test_data, features_type != 'fzn2feat', is_wlc='wlc' in features_type, hyperparam=None, use_pca=args.pca)
    elif model == 'svc':
        res = train_and_test_svc(train_data, test_data, features_type != 'fzn2feat', hyperparam=None, use_pca=args.pca)
    elif model == 'nn':
        res = train_and_test_nn_torch(train_data, test_data, None, use_pca=args.pca)
    else:
        raise Exception(f'Unsupported model type {model}')

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'Results successfully saved to {output_file}')

if __name__ == '__main__':
    env_8 = dict(os.environ, JULIA_NUM_THREADS="8")
    server_8 = subprocess.Popen(['ZincToWl', "--server", "/tmp/zinctowl_8.sock"], env=env_8, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    try:
        main()
    finally:
        server_8.terminate()
