import numpy as np
import argparse
from typing import Literal
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.graph_loader import load_graph
from common.wl_algorithms import wl_features, wl_extended_features
from tqdm import tqdm
from train_p_forest import train_and_test_rnd_forest
from train_p_svc import train_and_test_svc
import json
from copy import deepcopy
from sklearn.model_selection import train_test_split
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--features', type=str, required=True, choices=['wlc-0', 'wlc-1', 'wlc-2', 'wl-0', 'wl-1', 'wl-2', 'wln-0', 'wln-1', 'wln-2', 'wle-0', 'wle-1', 'wle-2', 'wlne-0', 'wlne-1', 'wlne-2', 'fzn2feat'])
parser.add_argument('-m', '--model', type=str, required=True, choices=['rnd-forest', 'svc'])
parser.add_argument('-r', '--rnd-state', type=int, required=True)
parser.add_argument('--result', required=True, type=str)

def load_data() -> list[dict]:
    '''
    loads the algorithm selection dataset. Contains, for each datapoint:
        - the model name
        - the instance name
        - the correct label (0: chuffed better, 1: cp-sat better, 2: they are the same)
        - solving time of the chuffed solver
        - solving time of the cp-sat solver
        - the path of the corresponding graph
    '''
    data = pd.read_csv('./data/algorithm_selection_dataset.csv')

    dict_data = []
    for i in range(len(data)):
        d = data.iloc[i]
        dict_data.append({'model': d['model'],
         'name': d['name'],
         'label': d['label'],
         'chuffed': d['chuffed'],
         'cp-sat': d['cp-sat'],
         'graph': './data/' + d['graph']}
        )

    return dict_data

def prune(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict],list[dict]]:
    train_features = np.array([t['features'] for t in train_data])
    magnitude = np.sum(train_features, axis=0)
    idxs, = np.where(magnitude <= 0)
    for t in train_data:
        t['features'] = np.array(np.delete(t['features'], idxs).tolist() + [np.sum(np.array(t['features'])[idxs])])
    for t in test_data:
        t['features'] = np.array(np.delete(t['features'], idxs).tolist() + [np.sum(np.array(t['features'])[idxs])])
    return train_data, test_data

def compute_wl_features(train_data:list[dict], test_data:list[dict], wl_type:Literal['standard','node_features','edge_features','node_edge_features'], max_iter:int) -> tuple[list[dict],list[dict]]:
    colors = {}
    MAX_COLORS = None

    for t in tqdm(train_data, desc='train data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        res = wl_features(g, colors, wl_type=wl_type, max_iter=max_iter, training=True, max_colors=MAX_COLORS, with_neighbours=False)
        t['features'] = res

    colors_names = set(sorted(set(int(c) for c in colors.values())))
    for t in train_data:
        res = t['features']
        counter = Counter(res)
        n = len(res)
        features = [counter.get(color, 0) / n for color in colors_names]
        t['features'] = features

    for t in tqdm(test_data, desc='test data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        res = wl_features(g, colors, wl_type=wl_type, max_iter=max_iter, training=False, max_colors=MAX_COLORS, with_neighbours=False)
        counter = Counter(res)
        n = len(res)
        features = [counter.get(color, 0) / n for color in colors_names]
        t['features'] = features

    return prune(train_data, test_data)

def compute_custom_wl(train_data:list[dict], test_data:list[dict], max_iter:int) -> tuple[list[dict],list[dict]]:
    colors = {}

    g_pairs = set()
    for t in tqdm(train_data, desc='train data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        res, extra = wl_extended_features(g, colors, max_iter=max_iter, training=True)
        t['features'] = res
        t['extra'] = extra
        for pair in extra['globals_pairs'].keys():
            g_pairs.add(pair)

    g_pairs = sorted(g_pairs)

    colors_names = set(sorted(set(int(c) for c in colors.values())))
    for t in train_data:
        res = t['features']
        counter = Counter(res)
        features = np.array([counter.get(color, 0) / t['extra']['n_nodes'] for color in colors_names])
        tot_pairs = max(sum(t['extra']['globals_pairs'].values()), 1)
        t['features'] = features.tolist() + [t['extra']['globals_pairs'].get(p,0)/tot_pairs for p in g_pairs] + [t['extra']['cpv'], t['extra']['cpp']]

    for t in tqdm(test_data, desc='test data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        res, extra = wl_extended_features(g, colors, max_iter=max_iter, training=False)
        counter = Counter(res)
        features = np.array([counter.get(color, 0) / extra['n_nodes'] for color in colors_names])
        tot_pairs = max(sum(extra['globals_pairs'].values()), 1)
        t['features'] = features.tolist() + [extra['globals_pairs'].get(p,0)/tot_pairs for p in g_pairs] + [extra['cpv'], extra['cpp']]

    return prune(train_data, test_data)

def get_fzn2feat(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict],list[dict]]:
    fzn2feat_features = pd.read_csv('./data/fzn2feat.csv')
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
        features_type:Literal['wlc-0', 'wlc-1', 'wlc-2', 'wl-0', 'wl-1', 'wl-2', 'wln-0', 'wln-1', 'wln-2', 'wln-0', 'wle-0', 'wle-1', 'wle-2', 'wlne-0', 'wlne-1', 'wlne-2', 'fzn2feat']
    ) -> tuple[list[dict], list[dict]]:
    '''
    for each feature-type returns the modified train and dataset agumented with the corresponding features
    '''

    if features_type == 'wl-0':
        return compute_wl_features(train_data, test_data, 'standard', 0)
    elif features_type == 'wl-1':
        return compute_wl_features(train_data, test_data, 'standard', 1)
    elif features_type == 'wl-2':
        return compute_wl_features(train_data, test_data, 'standard', 2)

    elif features_type == 'wln-0':
        return compute_wl_features(train_data, test_data, 'node_features', 0)
    elif features_type == 'wln-1':
        return compute_wl_features(train_data, test_data, 'node_features', 1)
    elif features_type == 'wln-2':
        return compute_wl_features(train_data, test_data, 'node_features', 2)

    elif features_type == 'wle-0':
        return compute_wl_features(train_data, test_data, 'edge_features', 0)
    elif features_type == 'wle-1':
        return compute_wl_features(train_data, test_data, 'edge_features', 1)
    elif features_type == 'wle-2':
        return compute_wl_features(train_data, test_data, 'edge_features', 2)

    elif features_type == 'wlne-0':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 0)
    elif features_type == 'wlne-1':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 1)
    elif features_type == 'wlne-2':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 2)

    elif features_type == 'wlc-0':
        return compute_custom_wl(train_data, test_data, 0)
    elif features_type == 'wlc-1':
        return compute_custom_wl(train_data, test_data, 1)
    elif features_type == 'wlc-2':
        return compute_custom_wl(train_data, test_data, 2)

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

def main():
    args = parser.parse_args()
    features_type:str = args.features
    rnd_state = args.rnd_state
    output_file:str = args.result
    model:Literal['rnd-forest', 'svc'] = args.model

    data = load_data()
    print('data loaded, starting to compute features')

    problems = list(set([d['model'] for d in data]))

    train_problems, test_problems = train_test_split(problems, test_size=.2, random_state=rnd_state)
    train_data = [d for d in data if d['model'] in train_problems]
    test_data = [d for d in data if d['model'] in test_problems]


    #data preparation, decide features type and pruning
    train_data, test_data = get_features(train_data, test_data, features_type)
    print('computed features, starting to train model')

    if model == 'rnd-forest':
        res = train_and_test_rnd_forest(train_data, test_data, 'wl' in features_type, 'wlc' in features_type)
    elif model == 'svc':
        res = train_and_test_svc(train_data, test_data, 'wl' in features_type, 'wlc' in features_type)
    with open(output_file, 'w') as f:
        json.dump(res, f)

if __name__ == '__main__':
    main()
