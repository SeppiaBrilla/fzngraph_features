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
import numpy as np
from feature_extraction import get_features
import subprocess

import warnings
warnings.filterwarnings("ignore")

import random
random.seed(42)
np.random.seed(42)

DEFAULT_PORTFOLIO = ['choco', 'chuffed', 'cp-sat', 'gecode', 'gurobi']
DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), '../../data/data_sequential_5solvers.csv')


def load_data(data_path: str = DEFAULT_DATASET) -> tuple[pd.DataFrame, list[str]]:
    """
    Loads the MiniZinc Challenge algorithm selection dataset.
    """
    data = pd.read_csv(data_path)
    solvers = [s for s in DEFAULT_PORTFOLIO if s in data.columns]
    if not solvers:
        solvers = DEFAULT_PORTFOLIO
    return data, solvers


def data_to_list(data: pd.DataFrame, solvers: list[str]) -> list[dict]:
    dict_data = []

    for i in range(len(data)):
        d = data.iloc[i]
        scores = [float(d[s]) for s in solvers]
        best_idx = int(np.argmax(scores))
        best_solver = solvers[best_idx]

        inst = {
            'model': str(d.get('problem', '')),
            'name': str(d.get('instance', '')),
            'year': int(d.get('year', 0)),
            'label': best_idx,
            'best_solver': best_solver,
            'graph': './data/graphs/' + str(d.get('graph', '')),
            'flatzinc': None,
            'solvers': solvers,
        }
        for s in solvers:
            inst[s] = float(d[s])
        dict_data.append(inst)

    return dict_data


def split_stratified_by_gap(df: pd.DataFrame, rnd_state: int, current_fold: int, max_fold: int, solvers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert max_fold > 0, f'max fold must be > 0. got {max_fold}'
    assert current_fold >= 0, f'current fold must positive. got {current_fold}'
    assert current_fold < max_fold, f'current fold must be < max fold. got current fold:{current_fold} and max fold: {max_fold}'

    df_copy = df.copy()
    df_copy['gap'] = df_copy[solvers].max(axis=1) - df_copy[solvers].min(axis=1)
    df_copy['gap_strata'] = pd.qcut(df_copy['gap'], q=5, labels=False, duplicates='drop')

    skf = StratifiedKFold(n_splits=max_fold, shuffle=True, random_state=rnd_state)
    idxs = [(train_idx, test_idx) for (train_idx, test_idx) in skf.split(df_copy, df_copy['gap_strata'])]
    train_idx, test_idx = idxs[current_fold]
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    return train_df, test_df


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--features', type=str, required=False, choices=[
    'wlce-1', 'wlce-2', 'wlc-1', 'wlc-2', 'wlcu-1', 'wlcu-2', 'wlceu-1', 'wlceu-2',
    'wl-1', 'wl-2', 'wln-1', 'wln-2', 'wlun-1', 'wlun-2', 'wle-0', 'wle-1', 'wle-2',
    'wlne-1', 'wlne-2', 'wlune-1', 'wlune-2', 'fzn2feat'
])
parser.add_argument('-m', '--model', type=str, required=False, choices=['svc', 'rnd-forest', 'nn', 'gb'])
parser.add_argument('--cv-fold', required=False, type=int)
parser.add_argument('--max-cv', required=False, type=int, default=5)
parser.add_argument('--result', required=False, type=str)
parser.add_argument('--rnd-state', required=False, type=int, default=42)
parser.add_argument('--pca', required=False, action='store_true', default=True, help='Apply PCA to reduce features')
parser.add_argument('--all-levels', action='store_true', help='Use all aggregation levels')
parser.add_argument('--year', type=int, default=None, help='Filter by challenge year (e.g. 2026)')
parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET, help='Path to dataset CSV')
parser.add_argument('--cache-dir', type=str, default='data/mzn_5solvers_features_cache', help='Feature cache directory')
parser.add_argument('--run-all-years', action='store_true', help='Run full evaluation over all years and save standard summary json files')


def evaluate_single_fold(
    features_type: str,
    model: str,
    fold: int,
    max_cv: int,
    rnd_state: int,
    dataset_path: str,
    year: int | None,
    output_file: str | None,
    pca: bool = True,
    all_levels: bool = False,
    cache_dir: str = 'data/mzn_5solvers_features_cache'
) -> dict:
    data, solvers = load_data(dataset_path)
    if year is not None:
        data = data[data['year'] == year].copy().reset_index(drop=True)
    print(f'Loaded MiniZinc dataset with {len(data)} instances (year={year}), starting feature preparation')

    train_df, test_df = split_stratified_by_gap(
        df=data,
        rnd_state=rnd_state,
        current_fold=fold,
        max_fold=max_cv,
        solvers=solvers
    )

    train_data = data_to_list(train_df, solvers)
    test_data = data_to_list(test_df, solvers)

    os.makedirs(cache_dir, exist_ok=True)
    year_prefix = f"{year}_" if year is not None else ""
    feat_cache_file = os.path.join(cache_dir, f"{year_prefix}{features_type}_fold{fold}.csv")

    train_data, test_data = get_features(train_data, test_data, features_type, all_levels, feat_cache_file)
    print(f'Prepared {features_type} features ({len(train_data[0]["features"])} dims), starting to train {model}')

    if model == 'rnd-forest':
        res = train_and_test_rnd_forest(train_data, test_data, is_wl=features_type != 'fzn2feat', is_wlc='wlc' in features_type, hyperparam=None, use_pca=pca)
    elif model == 'svc':
        res = train_and_test_svc(train_data, test_data, is_wl=features_type != 'fzn2feat', hyperparam=None, use_pca=pca)
    elif model == 'nn':
        res = train_and_test_nn_torch(train_data, test_data, None, use_pca=pca)
    else:
        raise Exception(f'Unsupported model type {model}')

    if output_file:
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'Results successfully saved to {output_file}')

    return res


def run_benchmark(target_years=None, results_dir='data/as-score-results/mzn_challenge', cache_dir='data/mzn_5solvers_features_cache'):
    """
    Evaluates MiniZinc Challenge algorithm selection across challenge years and saves
    per-year summary files matching generate_statistical_report expectations.
    """
    if target_years is None:
        target_years = list(range(2012, 2027))

    os.makedirs(results_dir, exist_ok=True)
    feature_types = ['wlc-1', 'wlc-2', 'wlce-1', 'wlce-2', 'fzn2feat']
    models = ['rnd-forest', 'svc', 'nn']

    for y in target_years:
        for feat_name in feature_types:
            for model_name in models:
                res_file = os.path.join(results_dir, f"{y}_{feat_name}_{model_name}.json")
                if os.path.exists(res_file):
                    print(f"Year {y} - {feat_name} - {model_name} already exists. Skipping.")
                    continue

                print(f"\n=======================================================")
                print(f"Running MZN Challenge Year {y} | {feat_name} | {model_name}")
                print(f"=======================================================")
                t_combo = time.time()
                fold_results = []

                for fold in range(5):
                    fold_res = evaluate_single_fold(
                        features_type=feat_name,
                        model=model_name,
                        fold=fold,
                        max_cv=5,
                        rnd_state=42,
                        dataset_path=DEFAULT_DATASET,
                        year=y,
                        output_file=None,
                        pca=True,
                        cache_dir=cache_dir
                    )
                    fold_res['fold'] = fold
                    fold_results.append(fold_res)

                norm_scores = [fr['norm_score'] for fr in fold_results]
                accuracies = [fr['accuracy'] for fr in fold_results]
                raw_scores = [fr['clf_score'] for fr in fold_results]
                sbs_scores = [fr['sbs_score'] for fr in fold_results]
                vbs_scores = [fr['vbs_score'] for fr in fold_results]

                tot_clf = float(sum(raw_scores))
                tot_sbs = float(sum(sbs_scores))
                tot_vbs = float(sum(vbs_scores))
                gap = tot_vbs - tot_sbs
                pooled_norm = (tot_clf - tot_sbs) / gap if gap > 0 else 0.0

                summary = {
                    'year': int(y),
                    'features': feat_name,
                    'model': model_name,
                    'norm_mean': float(np.mean(norm_scores)),
                    'norm_std': float(np.std(norm_scores)),
                    'acc_mean': float(np.mean(accuracies)),
                    'acc_std': float(np.std(accuracies)),
                    'raw_mean': float(np.mean(raw_scores)),
                    'raw_std': float(np.std(raw_scores)),
                    'folds': fold_results,
                    'elapsed_seconds': time.time() - t_combo,
                    'pooled_norm_score': pooled_norm,
                    'pct_vbs': (tot_clf / tot_vbs * 100.0) if tot_vbs > 0 else 0.0,
                    'total_clf_score': tot_clf,
                    'total_sbs_score': tot_sbs,
                    'total_vbs_score': tot_vbs,
                }

                with open(res_file, 'w') as fp:
                    json.dump(summary, fp, indent=2)
                print(f"Year {y} {feat_name} {model_name} -> Norm: {summary['norm_mean']:.4f} (Pooled: {pooled_norm:.4f})")


def main():
    args = parser.parse_args()

    if args.run_all_years:
        years = [args.year] if args.year is not None else list(range(2012, 2027))
        run_benchmark(target_years=years, cache_dir=args.cache_dir)
        return

    if not args.features or not args.model or args.cv_fold is None:
        parser.error("-f/--features, -m/--model, and --cv-fold are required when not running --run-all-years")

    evaluate_single_fold(
        features_type=args.features,
        model=args.model,
        fold=args.cv_fold,
        max_cv=args.max_cv,
        rnd_state=args.rnd_state,
        dataset_path=args.dataset,
        year=args.year,
        output_file=args.result,
        pca=args.pca,
        all_levels=args.all_levels,
        cache_dir=args.cache_dir
    )


if __name__ == '__main__':
    ztw_bin = os.environ.get("ZINC_TO_WL_BIN", "ZincToWl")
    env_8 = dict(os.environ, JULIA_NUM_THREADS="8")
    socket_path = "/tmp/zinctowl_mzn.sock"
    if os.path.exists(socket_path):
        os.remove(socket_path)
    server_8 = subprocess.Popen([ztw_bin, "--server", socket_path], env=env_8, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    try:
        main()
    finally:
        server_8.terminate()
