import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from smac import HyperparameterOptimizationFacade, Scenario
import argparse
from typing import Literal
import pandas as pd
from graph_loader import load_grap
from wl_algorithms import wl_features
from tqdm import tqdm
import json
from copy import deepcopy
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split
from collections import Counter

class CrossValidator:
    def __init__(self, train_data:list[dict], cv:int=5) -> None:
        self.cv = cv
        self.train_folds:list[tuple[tuple[np.ndarray, np.ndarray],tuple[np.ndarray, np.ndarray]]] = self.__split_data(train_data, cv)

    def __split_data(self, train_data:list[dict], cv:int) -> list[tuple[tuple[np.ndarray, np.ndarray],tuple[np.ndarray, np.ndarray]]]:
        problems = list(set([t['model'] for t in train_data]))
        n_elements = len(problems) // cv
        train_folds = []
        for c in range(cv):
            val_probs = problems[c * n_elements: (c+1) * n_elements]
            train_probs = problems[:c * n_elements] + problems[(c+1) * n_elements:]
            X_train = np.array([t['features'] for t in train_data if t['model'] in train_probs])
            y_train = np.array([t['label'] for t in train_data if t['model'] in train_probs])
            X_val = np.array([t['features'] for t in train_data if t['model'] in val_probs])
            y_val = np.array([t['label'] for t in train_data if t['model'] in val_probs])

            train_folds.append(((X_train, y_train), (X_val, y_val)))

        return train_folds

    def score(self, clf:RandomForestClassifier) -> np.ndarray:
        scores = []
        for (X_train, y_train), (X_val, y_val) in self.train_folds:
            _clf = clone(clf)
            _clf.fit(X_train, y_train)
            scores.append(accuracy_score(y_val, _clf.predict(X_val)))

        return np.array(scores)

def train_rnd_forest(train_data:list[dict]) -> dict:

    #parameters: https://www.researchgate.net/figure/Tested-parameter-grid-for-random-forest-classifier_tbl1_350998771
    n_estimators = Integer('n_estimators', (200, 1001), default=200)
    criterion = Categorical('criterion', ['gini', 'entropy', 'log_loss'], default='gini')
    max_features = Categorical('max_features', ['sqrt', 'log2', None])
    max_depth = Integer('max_depth', (9, 101))
    min_sample_split = Integer('min_samples_split', (2, 10), default=5)
    min_sample_leaf = Integer('min_samples_leaf', (1, 10), default=2)

    cs = ConfigurationSpace(seed=42)
    hyprparam = [n_estimators, criterion, max_features, max_depth, min_sample_split, min_sample_leaf]#, k_features]
    cs.add(hyprparam)

    scenario = Scenario(
        cs,
        n_workers=-1,
        walltime_limit=20*60,
        n_trials=10000000
    )

    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)

    validator = CrossValidator(train_data, cv=5)

    def train(config:Configuration, seed:int=42) -> float:
        config_dict = dict(config)
        if config_dict['max_depth'] == 9:
            config_dict['max_depth'] = None

        clf = RandomForestClassifier(**config_dict, random_state=seed)
        scores = validator.score(clf) #cross_val_score(clf, X_train, y_train, cv=5)
        return float(np.mean(1 - scores))

    smac = HyperparameterOptimizationFacade(
        scenario,
        train,
        initial_design=initial_design,
        overwrite=True,
    )
    incumbent = smac.optimize()
    if isinstance(incumbent, list):
        incumbent = incumbent[0]
    config_dict = dict(incumbent)
    if config_dict['max_depth'] == 9:
        config_dict['max_depth'] = None
    return config_dict

def test_rnd_forest(clf:RandomForestClassifier, X_test:np.ndarray, y_test:np.ndarray, hyperparam:dict) -> dict:
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f'accuracy: {accuracy:.3f}')

    return {
        'accuracy': float(accuracy),
        'hyperparameters': hyperparam
        }

def train_and_test_rnd_forest(train_data:list[dict], test_data:list[dict]) -> dict:
    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    hyperparam = train_rnd_forest(train_data)

    # k = hyperparam['k']
    # del hyperparam['k']

    clf = RandomForestClassifier(**hyperparam, random_state=42)
    print(np.mean(cross_val_score(clf, X_train, y_train, cv=5)))

    # red = Preprocesser(k)
    # red.fit(X_train)
    # X_train = red.transform(X_train)
    # X_test = red.transform(X_test)

    clf.fit(X_train, y_train)
    # hyperparam['k'] = k
    return test_rnd_forest(clf, X_test, y_test, hyperparam)


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--features', type=str, required=True, choices=['wl-3', 'wl-5', 'wl-7', 'wln-3', 'wln-5', 'wln-7', 'wle-3', 'wle-5', 'wle-7', 'wlne-3', 'wlne-5', 'wlne-7', 'fzn2feat'])
parser.add_argument('-r', '--rnd-state', type=int, required=True)
# parser.add_argument('-m', '--model', type=str, required=True, choices=['kmeans', 'rnd-forest', 'nn'])
# parser.add_argument('--cv-fold', required=True, type=int, choices=[0,1,2,3,4])
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
    data = pd.read_csv('./parallelise_cplex.csv')

    dict_data = []
    for i in range(len(data)):
        d = data.iloc[i]
        dict_data.append({'model': d['model'],
         'name': d['name'],
         'label': d['y'],
         'graph': d['graph']}
        )

    return dict_data

def prune(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict],list[dict]]:
    train_features = np.array([t['features'] for t in train_data])
    magnitude = np.sum(train_features, axis=0)
    idxs, = np.where(magnitude <= 20)
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
                g = load_grap(f)
            res = wl_features(g, colors, wl_type=wl_type, max_iter=max_iter, training=True, max_colors=MAX_COLORS, with_neighbours=False)
            t['features'] = res

    colors_names = sorted(set(int(c) for c in colors.values()))

    for t in train_data:
        res = t['features']
        counter = Counter(res)
        features = [counter.get(color, 0) for color in colors_names]
        t['features'] = features

    for t in tqdm(test_data, desc='test data'):
        with open(t['graph']) as f:
            g = load_grap(f)
        res = wl_features(g, colors, wl_type=wl_type, max_iter=max_iter, training=False, max_colors=MAX_COLORS, with_neighbours=False)
        counter = Counter(res)
        features = [counter.get(color, 0) for color in colors_names]
        t['features'] = features

    return prune(train_data, test_data)

def get_fzn2feat(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict],list[dict]]:
    fzn2feat_features = pd.read_csv('./fzn2feat.csv')
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
        features_type:Literal['wl-3', 'wl-5', 'wl-7', 'wln-3', 'wln-5', 'wln-7', 'wle-3', 'wle-5', 'wle-7', 'wlne-3', 'wlne-5', 'wlne-7', 'fzn2feat']
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
    # model:str = args.model
    # fold:int = args.cv_fold
    rnd_state = args.rnd_state
    output_file:str = args.result

    data = load_data()
    print('data loaded, starting to compute features')

    problems = list(set([d['model'] for d in data]))

    train_problems, test_problems = train_test_split(problems, test_size=.2, random_state=rnd_state)
    train_data = [d for d in data if d['model'] in train_problems]
    test_data = [d for d in data if d['model'] in test_problems]


    #data preparation, decide features type and pruning
    train_data, test_data = get_features(train_data, test_data, features_type)
    print('computed features, starting to train model')

    # if model == 'rnd-forest':
    res = train_and_test_rnd_forest(train_data, test_data)
        # res = train_and_test_rnd_forest_forward_selector(train_data, test_data)
    # elif model == 'kmeans':
    #     res = train_and_test_kmeans(train_data, test_data, features_type != 'fzn2feat')
    # elif model == 'nn':
    #     res = train_and_test_nn(train_data, test_data, False)
    # else:
    #     raise Exception(f'still unsupported model type {model}')
    with open(output_file, 'w') as f:
        json.dump(res, f)

if __name__ == '__main__':
    main()
