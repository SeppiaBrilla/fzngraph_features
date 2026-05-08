import argparse
from typing import Literal
import pandas as pd
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.graph_loader import load_graph
from train_neural_network import train_and_test_nn
from train_torch_neural_network import train_and_test_nn_torch
from sklearn.model_selection import StratifiedKFold
from common.wl_algorithms import wl_features, wl_extended_features, wl_extended_features_with_edges, undirected_wl_extended_features, undirected_wl_extended_features_with_edges, undirected_wl_with_node_and_edge_features, undirected_wl_with_node_features
from train_as_forest import train_and_test_rnd_forest
from train_as_svc import train_and_test_svc
from train_as_knn import train_and_test_knn
from train_as_forward_knn import train_and_test_forward_knn
from copy import deepcopy
import numpy as np
from collections import Counter
from tqdm import tqdm

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

def prune(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict],list[dict]]:
    train_features = np.array([t['features'] for t in train_data])
    magnitude = np.sum(train_features, axis=0)
    idxs, = np.where(magnitude <= 0)
    for t in train_data:
        t['features'] = np.array(np.delete(t['features'], idxs).tolist() + [np.sum(np.array(t['features'])[idxs])])
    for t in test_data:
        t['features'] = np.array(np.delete(t['features'], idxs).tolist() + [np.sum(np.array(t['features'])[idxs])])
    return train_data, test_data

def compute_wl_features(train_data:list[dict], test_data:list[dict], wl_type:Literal['standard','node_features','edge_features','node_edge_features'], max_iter:int, all_levels:bool, undirected=False) -> tuple[list[dict],list[dict]]:
    colors = {}
    MAX_COLORS = None

    for t in tqdm(train_data, desc='train data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        if undirected:
            if wl_type == 'node_features':
                res = undirected_wl_with_node_features(g, colors, max_iter, True, MAX_COLORS, with_neighbours=False)
            elif wl_type == 'node_edge_features':
                res = undirected_wl_with_node_and_edge_features(g, colors, max_iter, True, MAX_COLORS, with_neighbours=False)
            else:
                raise Exception(f'unsupported undirected type {wl_type}')
        else:
            res = wl_features(g, colors, wl_type=wl_type, max_iter=max_iter, training=True, max_colors=MAX_COLORS, with_neighbours=False)
        t['features'] = res

    colors_names = set(sorted(set(int(c) for c in colors.values())))
    for t in train_data:
        res = t['features']
        if all_levels:
            features = []
            for r in res:
                counter = Counter(r)
                n = len(r)
                features.extend([counter.get(color, 0) / n for color in colors_names])
        else:
            r = res[-1]
            counter = Counter(r)
            n = len(r)
            features = [counter.get(color, 0) / n for color in colors_names]
        t['features'] = features

    for t in tqdm(test_data, desc='test data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        if undirected:
            if wl_type == 'node_features':
                res = undirected_wl_with_node_features(g, colors, max_iter, False, MAX_COLORS, with_neighbours=False)
            elif wl_type == 'node_edge_features':
                res = undirected_wl_with_node_and_edge_features(g, colors, max_iter, False, MAX_COLORS, with_neighbours=False)
            else:
                raise Exception(f'unsupported undirected type {wl_type}')
        else:
            res = wl_features(g, colors, wl_type=wl_type, max_iter=max_iter, training=False, max_colors=MAX_COLORS, with_neighbours=False)
        if all_levels:
            features = []
            for r in res:
                counter = Counter(r)
                n = len(r)
                features.extend([counter.get(color, 0) / n for color in colors_names])
        else:
            r = res[-1]
            counter = Counter(r)
            n = len(r)
            features = [counter.get(color, 0) / n for color in colors_names]
        t['features'] = features

    return prune(train_data, test_data)

def compute_custom_wl(train_data:list[dict], test_data:list[dict], max_iter:int, edge:bool, undirected:bool, all_levels:bool) -> tuple[list[dict],list[dict]]:
    colors = {}

    g_pairs = set()
    for t in tqdm(train_data, desc='train data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        if undirected and not edge:
            res, extra = undirected_wl_extended_features(g, colors, max_iter=max_iter, training=True)
            res = [res]
        elif undirected and edge:
            res, extra = undirected_wl_extended_features_with_edges(g, colors, max_iter=max_iter, training=True)
            res = [res]
        elif not undirected and edge:
            res, extra = wl_extended_features_with_edges(g, colors, max_iter=max_iter, training=True)
        else:
            res, extra = wl_extended_features(g, colors, max_iter=max_iter, training=True)
        t['features'] = res
        t['extra'] = extra
        for pair in extra['globals_pairs'].keys():
            g_pairs.add(pair)

    g_pairs = sorted(g_pairs)

    colors_names = set(sorted(set(int(c) for c in colors.values())))
    for t in train_data:
        res = t['features']
        if all_levels:
            features = []
            for r in res:
                counter = Counter(r)
                features.extend([counter.get(color, 0) / t['extra']['n_nodes'] for color in colors_names])
            features = np.array(features)
        else:
            r = res[-1]
            counter = Counter(r)
            features = np.array([counter.get(color, 0) / t['extra']['n_nodes'] for color in colors_names])
        tot_pairs = max(sum(t['extra']['globals_pairs'].values()), 1)
        t['features'] = features.tolist() + [t['extra']['globals_pairs'].get(p,0)/tot_pairs for p in g_pairs] + [t['extra']['cpv'], t['extra']['cpp']]

    for t in tqdm(test_data, desc='test data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        if undirected and not edge:
            res, extra = undirected_wl_extended_features(g, colors, max_iter=max_iter, training=True)
            res = [res]
        elif undirected and edge:
            res, extra = undirected_wl_extended_features_with_edges(g, colors, max_iter=max_iter, training=True)
            res = [res]
        elif not undirected and edge:
            res, extra = wl_extended_features_with_edges(g, colors, max_iter=max_iter, training=True)
        else:
            res, extra = wl_extended_features(g, colors, max_iter=max_iter, training=True)
        
        if all_levels:
            features = []
            for r in res:
                counter = Counter(r)
                features.extend([counter.get(color, 0) / extra['n_nodes'] for color in colors_names])
            features = np.array(features)
        else:
            r = res[-1]
            counter = Counter(r)
            features = np.array([counter.get(color, 0) / extra['n_nodes'] for color in colors_names])
            
        tot_pairs = max(sum(extra['globals_pairs'].values()), 1)
        t['features'] = features.tolist() + [extra['globals_pairs'].get(p,0)/tot_pairs for p in g_pairs] + [extra['cpv'], extra['cpp']]

    return prune(train_data, test_data)

def get_fzn2feat(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict],list[dict]]:
    fzn2feat_features = pd.read_csv('./data/fzn2feat_joined.csv')
    for t in train_data:
        d = fzn2feat_features[(fzn2feat_features['problem'] == t['model']) & (fzn2feat_features['name'] == t['name'])]
        d = d.drop(columns=['problem','name'])
        assert len(d.values) == 1, f'not one element: {d.values}, {t}'
        t['features'] = d.values[0]

    for t in test_data:
        d = fzn2feat_features[(fzn2feat_features['problem'] == t['model']) & (fzn2feat_features['name'] == t['name'])]
        d = d.drop(columns=['problem','name'])
        assert len(d.values) == 1, f'not one element: {d.values}, {t}'
        t['features'] = d.values[0]

    return train_data, test_data

def get_features(
        train_data:list[dict],
        test_data:list[dict],
        features_type:Literal['wlce-1', 'wlce-2', 'wlc-1', 'wlc-2', 'wlcu-1', 'wlcu-2', 'wlceu-1', 'wlceu-2', 'wl-1', 'wl-2', 'wln-1', 'wlun-1', 'wlun-2', 'wlune-1', 'wlune-2', 'wln-2', 'wle-0', 'wle-1', 'wle-2', 'wlne-1', 'wlne-2', 'fzn2feat'],
        all_levels:bool=False
    ) -> tuple[list[dict], list[dict]]:
    '''
    for each feature-type returns the modified train and dataset agumented with the corresponding features
    '''

    if features_type == 'wl-1':
        return compute_wl_features(train_data, test_data, 'standard', 1, all_levels)
    elif features_type == 'wl-2':
        return compute_wl_features(train_data, test_data, 'standard', 2, all_levels)

    elif features_type == 'wln-1':
        return compute_wl_features(train_data, test_data, 'node_features', 1, all_levels)
    elif features_type == 'wln-2':
        return compute_wl_features(train_data, test_data, 'node_features', 2, all_levels)

    elif features_type == 'wle-1':
        return compute_wl_features(train_data, test_data, 'edge_features', 1, all_levels)
    elif features_type == 'wle-2':
        return compute_wl_features(train_data, test_data, 'edge_features', 2, all_levels)

    elif features_type == 'wlne-1':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 1, all_levels)
    elif features_type == 'wlne-2':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 2, all_levels)

    elif features_type == 'wlun-1':
        return compute_wl_features(train_data, test_data, 'node_features', 1, False, undirected=True)
    elif features_type == 'wlun-2':
        return compute_wl_features(train_data, test_data, 'node_features', 2, False, undirected=True)

    elif features_type == 'wlune-1':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 1, False, undirected=True)
    elif features_type == 'wlune-2':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 2, False, undirected=True)

    elif features_type == 'wlc-1':
        return compute_custom_wl(train_data, test_data, 1, False, False, all_levels)
    elif features_type == 'wlc-2':
        return compute_custom_wl(train_data, test_data, 2, False, False, all_levels)

    elif features_type == 'wlce-1':
        return compute_custom_wl(train_data, test_data, 1, True, False, all_levels)
    elif features_type == 'wlce-2':
        return compute_custom_wl(train_data, test_data, 2, True, False, all_levels)

    elif features_type == 'wlcu-1':
        return compute_custom_wl(train_data, test_data, 1, False, True, all_levels)
    elif features_type == 'wlcu-2':
        return compute_custom_wl(train_data, test_data, 2, False, True, all_levels)

    elif features_type == 'wlceu-1':
        return compute_custom_wl(train_data, test_data, 1, False, True, all_levels)
    elif features_type == 'wlceu-2':
        return compute_custom_wl(train_data, test_data, 2, False, True, all_levels)

    elif features_type == 'fzn2feat':
        return get_fzn2feat(train_data, test_data)

    raise Exception(f'unsupported features type {features_type}')

def split_data(data:list[dict]) -> tuple[list[dict],list[dict]]:
    test_models = {'tower', 'word_equations_02_track_8-int', 'Unit-Commitment', 'chessboard',
              'ctw', 'yumi-dynamic', 'community-detection', 'handball', 'TableLayout',
              'peaceable_queens_mznc2021', 'sudoku_fixed', 'mrcpsp', 'unison'}
    train_data = [deepcopy(d) for d in data if not d['model'] in test_models]
    test_data = [deepcopy(d) for d in data if d['model'] in test_models]
    print(len(train_data), len(test_data))
    return train_data, test_data

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--features', type=str, required=True, choices=['wlce-1', 'wlce-2', 'wlc-1', 'wlc-2', 'wlcu-1', 'wlcu-2', 'wlceu-1', 'wlceu-2', 'wl-1', 'wl-2', 'wln-1', 'wln-2', 'wlun-1', 'wlun-2', 'wle-0', 'wle-1', 'wle-2', 'wlne-1', 'wlne-2', 'wlune-1', 'wlune-2', 'fzn2feat'])
parser.add_argument('-m', '--model', type=str, required=True, choices=['svc', 'rnd-forest', 'nn'])
parser.add_argument('--cv-fold', required=True, type=int, choices=[0,1,2,3,4])
parser.add_argument('--max-cv', required=True, type=int)
parser.add_argument('--result', required=True, type=str)
parser.add_argument('--rnd-state', required=True, type=int)
parser.add_argument('--all-levels', action='store_true', help='Use a concatenation of all aggregation levels instead of only the last one')

def main():
    args = parser.parse_args()
    features_type:Literal['wlce-1', 'wlce-2', 'wlc-1', 'wlcu-1', 'wlcu-2', 'wlceu-1', 'wlceu-2', 'wlc-2', 'wl-1',
                          'wl-2', 'wln-1', 'wln-2', 'wle-1', 'wle-2', 
                          'wlne-1', 'wlne-2', 'fzn2feat'] = args.features
    model:str = args.model
    fold:int = args.cv_fold
    output_file:str = args.result
    max_cv:int = args.max_cv
    rnd_state:int = args.rnd_state
    all_levels:bool = args.all_levels
    # train_time:None|int = args.train_time

    data = load_data()
    print('data loaded, starting to compute features')

    train_data, test_data = split_stratified_by_gap(df=data,
                                                    rnd_state=rnd_state,
                                                    current_fold=fold,
                                                    max_fold=max_cv)

    train_data, test_data = data_to_list(train_data), data_to_list(test_data)

    #data preparation, decide features type and pruning
    train_data, test_data = get_features(train_data, test_data, features_type, all_levels)
    print('computed features, starting to train model')

    if model == 'rnd-forest':
        res = train_and_test_rnd_forest(train_data, test_data, features_type != 'fzn2feat', is_wlc= 'wlc' in features_type)
    elif model == 'svc':
        res = train_and_test_svc(train_data, test_data, features_type != 'fzn2feat')
    elif model == 'nn':
        res = train_and_test_nn_torch(train_data, test_data, False)
    else:
        raise Exception(f'still unsupported model type {model}')
    with open(output_file, 'w') as f:
        json.dump(res, f)

if __name__ == '__main__':
    main()
