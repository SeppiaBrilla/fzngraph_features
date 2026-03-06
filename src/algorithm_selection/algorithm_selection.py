import argparse
from typing import Literal
import pandas as pd
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.graph_loader import load_graph
from train_neural_network import train_and_test_nn
from common.wl_algorithms import wl_features
from train_as_forest import train_and_test_rnd_forest
from train_as_kmeans import train_and_test_kmeans
from copy import deepcopy
import numpy as np
from collections import Counter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--features', type=str, required=True, choices=['wl-3', 'wl-5', 'wl-7', 'wln-1', 'wln-2', 'wln-3', 'wln-5', 'wln-7', 'wle-3', 'wle-5', 'wle-7', 'wlne-3', 'wlne-5', 'wlne-7', 'fzn2feat'])
parser.add_argument('-m', '--model', type=str, required=True, choices=['kmeans', 'rnd-forest', 'nn'])
parser.add_argument('--cv-fold', required=True, type=int, choices=[0,1,2,3,4])
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
        features = [counter.get(color, 0) for color in colors_names]
        t['features'] = features

    for t in tqdm(test_data, desc='test data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        res = wl_features(g, colors, wl_type=wl_type, max_iter=max_iter, training=False, max_colors=MAX_COLORS, with_neighbours=False)
        counter = Counter(res)
        features = [counter.get(color, 0) for color in colors_names]
        t['features'] = features

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
        features_type:Literal['wl-3', 'wl-5', 'wl-7', 'wln-1', 'wln-2', 'wln-3', 'wln-5', 'wln-7', 'wle-3', 'wle-5', 'wle-7', 'wlne-3', 'wlne-5', 'wlne-7', 'fzn2feat']
    ) -> tuple[list[dict], list[dict]]:
    '''
    for each feature-type returns the modified train and dataset agumented with the corresponding features
    '''

    if features_type == 'wl-3':
        return compute_wl_features(train_data, test_data, 'standard', 3)
    elif features_type == 'wl-5':
        return compute_wl_features(train_data, test_data, 'standard', 5)
    elif features_type == 'wl-7':
        return compute_wl_features(train_data, test_data, 'standard', 7)

    elif features_type == 'wln-1':
        return compute_wl_features(train_data, test_data, 'node_features', 1)
    elif features_type == 'wln-2':
        return compute_wl_features(train_data, test_data, 'node_features', 2)
    elif features_type == 'wln-3':
        return compute_wl_features(train_data, test_data, 'node_features', 3)
    elif features_type == 'wln-5':
        return compute_wl_features(train_data, test_data, 'node_features', 5)
    elif features_type == 'wln-7':
        return compute_wl_features(train_data, test_data, 'node_features', 7)

    elif features_type == 'wle-3':
        return compute_wl_features(train_data, test_data, 'edge_features', 3)
    elif features_type == 'wle-5':
        return compute_wl_features(train_data, test_data, 'edge_features', 5)
    elif features_type == 'wle-7':
        return compute_wl_features(train_data, test_data, 'edge_features', 7)

    elif features_type == 'wlne-3':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 3)
    elif features_type == 'wlne-5':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 5)
    elif features_type == 'wlne-7':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 7)

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
    model:str = args.model
    fold:int = args.cv_fold
    output_file:str = args.result

    data = load_data()
    print('data loaded, starting to compute features')

    MAX_SPLITS = 5 #number of folds
    data_per_split = len(data) // MAX_SPLITS #number of elements per fold. At each training step 4 folds are used for training and 1 for test
    test_data = data[data_per_split*fold: data_per_split *(fold+1)] 
    train_data = data[:data_per_split*fold] + data[data_per_split *(fold+1):]
    # train_data, test_data = split_data(data)

    #data preparation, decide features type and pruning
    train_data, test_data = get_features(train_data, test_data, features_type)
    print('computed features, starting to train model')

    if model == 'rnd-forest':
        res = train_and_test_rnd_forest(train_data, test_data)
        # res = train_and_test_rnd_forest_forward_selector(train_data, test_data)
    elif model == 'kmeans':
        res = train_and_test_kmeans(train_data, test_data, features_type != 'fzn2feat')
    elif model == 'nn':
        res = train_and_test_nn(train_data, test_data, False)
    else:
        raise Exception(f'still unsupported model type {model}')
    with open(output_file, 'w') as f:
        json.dump(res, f)

if __name__ == '__main__':
    main()
