import argparse
from typing import Literal
import pandas as pd
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train_torch_neural_network import train_and_test_nn_torch
from sklearn.model_selection import StratifiedKFold
from train_as_forest import train_and_test_rnd_forest
from train_as_svc import train_and_test_svc
# from train_as_gradient_boosting import train_and_test_gradient_boosting
from copy import deepcopy
import numpy as np
from common.feature_extraction import get_features

import random
random.seed(42)
np.random.seed(42)

def load_data() -> pd.DataFrame: #list[dict]:
    '''
    loads the algorithm selection dataset. Contains, for each datapoint:
        - the model name
        - the instance name
        - the correct label (0: chuffed better, 1: cp-sat better, 2: they are the same)
        - solving time of the chuffed solver
        - solving time of the cp-sat solver
        - the path of the corresponding graph
    '''
    data = pd.read_csv('./data/algorithm_selection_dataset_score.csv')
    return data

def data_to_list(data:pd.DataFrame) -> list[dict]:
    dict_data = []
    for i in range(len(data)):
        d = data.iloc[i]
        dict_data.append({'model': d['model'],
         'name': d['name'],
         'label': d['label'],
         'chuffed': d['chuffed'],
         'cp-sat': d['cp-sat'],
         'cplex': d['cplex'],
         'graph': './data/' + d['graph']}
        )

    return dict_data

def split_stratified_by_gap(df:pd.DataFrame, rnd_state, current_fold:int, max_fold:int) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert max_fold > 0, f'max fold must be > 0. got {max_fold}'
    assert current_fold >= 0, f'current fold must positive. got {current_fold}'
    assert current_fold < max_fold, f'current fold must be < max fold. got current fold:{current_fold} and max fold: {max_fold}'

    solvers = df[['cp-sat', 'chuffed', 'cplex']]
    df['gap'] = solvers.max(axis=1) - solvers.min(axis=1)

    df['gap_strata'] = pd.qcut(df['gap'], q=5, labels=False, duplicates='drop')

    skf = StratifiedKFold(n_splits=max_fold, shuffle=True, random_state=rnd_state)
    idxs = [(train_idx, test_idx) for (train_idx, test_idx) in skf.split(df, df['gap_strata'])]
    train_idx, test_idx = idxs[current_fold]
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    for d in [train_df, test_df]:
        assert isinstance(d, pd.DataFrame), type(d)
        d.drop(columns=['gap', 'gap_strata'], inplace=True)

    assert isinstance(train_df, pd.DataFrame), type(train_df)
    assert isinstance(test_df, pd.DataFrame), type(test_df)

    return train_df, test_df

def split_data(data:list[dict]) -> tuple[list[dict],list[dict]]:
    test_models = {'tower', 'word_equations_02_track_8-int', 'Unit-Commitment', 'chessboard',
              'ctw', 'yumi-dynamic', 'community-detection', 'handball', 'TableLayout',
              'peaceable_queens_mznc2021', 'sudoku_fixed', 'mrcpsp', 'unison'}
    train_data = [deepcopy(d) for d in data if not d['model'] in test_models]
    test_data = [deepcopy(d) for d in data if d['model'] in test_models]
    print(len(train_data), len(test_data))
    return train_data, test_data

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--features', type=str, required=True, choices=['wlce-1', 'wlce-2', 'wlc-1', 'wlc-2', 'wlcu-1', 'wlcu-2', 'wlceu-1', 'wlceu-2', 'wl-1', 'wl-2', 'wln-1', 'wln-2', 'wlun-1', 'wlun-2', 'wle-0', 'wle-1', 'wle-2', 'wlne-1', 'wlne-2', 'wlune-1', 'wlune-2', 'fzn2feat', 'combined'])
parser.add_argument('-m', '--model', type=str, required=True, choices=['svc', 'rnd-forest', 'nn', 'gb'])
parser.add_argument('--cv-fold', required=True, type=int)
parser.add_argument('--max-cv', required=True, type=int)
parser.add_argument('--result', required=True, type=str)
parser.add_argument('--rnd-state', required=True, type=int)
parser.add_argument('--all-levels', action='store_true', help='Use a concatenation of all aggregation levels instead of only the last one')

def main():
    args = parser.parse_args()
    features_type:Literal['wlce-1', 'wlce-2', 'wlc-1', 'wlcu-1', 'wlcu-2', 'wlceu-1', 'wlceu-2', 'wlc-2', 'wl-1',
                          'wl-2', 'wln-1', 'wln-2', 'wle-1', 'wle-2',
                          'wlne-1', 'wlne-2', 'fzn2feat', 'combined'] = args.features
    model:str = args.model
    fold:int = args.cv_fold
    output_file:str = args.result
    max_cv:int = args.max_cv
    rnd_state:int = args.rnd_state
    all_levels:bool = args.all_levels

    data = load_data()
    print('data loaded, starting to compute features')

    train_data, test_data = split_stratified_by_gap(df=data,
                                                    rnd_state=rnd_state,
                                                    current_fold=fold,
                                                    max_fold=max_cv)


    train_data, test_data = data_to_list(train_data), data_to_list(test_data)
    # with open(output_file) as f:
    #         hyperparams = json.load(f)['hyperparameters']
    # if not 'size' in hyperparams:
    #     return

    #data preparation, decide features type and pruning
    train_data, test_data = get_features(train_data, test_data, features_type, all_levels, f'data/features/{features_type.replace("-","")}-{fold}-{rnd_state}.csv')
    print('computed features, starting to train model')

    if model == 'rnd-forest':
        # with open(output_file) as f:
        #     hyperparams = json.load(f)['hyperparameters']
        #     del hyperparams['size']
        hyperparams = None
        res = train_and_test_rnd_forest(train_data, test_data, features_type != 'fzn2feat', is_wlc= 'wlc' in features_type, hyperparam=hyperparams)
    elif model == 'svc':
        # with open(output_file) as f:
        #     hyperparams = json.load(f)['hyperparameters']
        #     print('loaded hyperparams')
        #     del hyperparams['size']
        hyperparams = None
        res = train_and_test_svc(train_data, test_data, features_type != 'fzn2feat', hyperparam=hyperparams)
    # elif model == 'gb':
    #     res = train_and_test_gradient_boosting(train_data, test_data)
    elif model == 'nn':
        # with open(output_file) as f:
        #     hyperparams = json.load(f)['hyperparameters']
        res = train_and_test_nn_torch(train_data, test_data, None)
    else:
        raise Exception(f'still unsupported model type {model}')
    with open(output_file, 'w') as f:
        json.dump(res, f)

if __name__ == '__main__':
    main()
