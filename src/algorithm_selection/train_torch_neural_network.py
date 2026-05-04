import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.decomposition import PCA
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random, math
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from common.torch_mlp import TorchMLPWrapper


def cross_val_score(clf, X:np.ndarray, y:np.ndarray, scores:np.ndarray, cv:int=5) -> float:
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    pred_scores = []
    quantiles = np.linspace(0, 100, 8)
    gap = np.abs(scores[:, 0] - scores[:, 1])
    bins = np.unique(np.percentile(gap, quantiles))
    buckets = np.digitize(gap, bins[1:-1])
    for train_idx, val_idx in kf.split(X, buckets):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        scores_val = scores[val_idx]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        pred_score = sum([scores_val[i,p] for i,p in enumerate(pred)])
        t0 = sum([scores_val[i,0] for i,_ in enumerate(pred)])
        t1 = sum([scores_val[i,1] for i,_ in enumerate(pred)])
        t2 = sum([scores_val[i,2] for i,_ in enumerate(pred)])
        sb_score = max(t1, t0, t2)

        pred_scores.append(pred_score/sb_score)

    return float(np.mean(pred_scores))

def _evaluate_combination(params: dict, X: np.ndarray, y: np.ndarray, scores:np.ndarray, use_gpu:bool=False) -> dict:
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    features = X.shape[1]
    layer_sizes = (features, features * 2, features, features // 2)

    device = 'cuda' if use_gpu else 'auto'
    model = TorchMLPWrapper(**params, hidden_layer_sizes=layer_sizes, device=device)
    score = cross_val_score(model, X, y, scores, cv=3)
    return {"params": params, "score": score}

def find_hyperparameters_nn_torch(
    X: np.ndarray,
    y: np.ndarray,
    scores:np.ndarray,
    n_jobs: int,
    use_gpu:bool=False
    ) -> dict:

    param_grid = {
        'activation': ['tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [15000],
        'random_state': [42]
    }

    all_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(all_combinations)

    n_workers = n_jobs

    worker_fn = partial(_evaluate_combination, X=X, y=y, scores=scores, use_gpu=use_gpu)

    # Run in parallel
    results = []
    with Pool(processes=n_workers) as pool:
        with tqdm(
            total=n_combinations,
            desc="hyperparameter search",
            unit="combo",
            dynamic_ncols=True,
        ) as pbar:
            for result in pool.imap(worker_fn, all_combinations):
                results.append(result)
                pbar.set_postfix(score=f"{result['score']:.4f}", refresh=False)
                pbar.update()
 
    rows = []
    for r in results:
        row = {'param':r["params"], "score": r["score"]}
        rows.append(row)

    best_score = max(rows, key=lambda x: x['score'])['score']
    equivalent_scores = [r for r in rows if math.isclose(r['score'], best_score, rel_tol=0.001)]
    best_config = min(equivalent_scores, key=lambda x: (x['param']['max_iter'], x['param']['alpha']))
    print('best config:', best_config)

    return best_config['param']

def size_evaluate_nn_torch(param:dict, hyperparams:dict, X:np.ndarray, y:np.ndarray, scores:np.ndarray) -> tuple[int|None,float]:
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    size = param['feature_size']

    if size is not None:
        pca = PCA(size, random_state=42)
        X_small = pca.fit_transform(X)
    else:
        X_small = X

    features = X_small.shape[1]
    layer_sizes = (features, max(1, features // 2), max(1, features // 4))

    clf = TorchMLPWrapper(**hyperparams, hidden_layer_sizes=layer_sizes)

    score = cross_val_score(clf, X_small, y, scores, 3)

    return size, score

def test_nn_torch(clf, X_test:np.ndarray, y_test:np.ndarray, test_data:list[dict], hyperparam:dict) -> dict:
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    pred_score = 0
    chuffed_score = 0
    cplex_score = 0
    cp_sat_score = 0
    vbs_score = 0
    predictions = {}
    for i, e in enumerate(test_data):
        x = np.array([X_test[i]])
        pred = clf.predict(x)[0]
        predictions[f"{e['model']}-sep-{e['name']}"] = int(pred)
        if pred == 0:
            pred_score += e['cp-sat']
        elif pred == 1:
            pred_score += e['chuffed']
        elif pred == 2:
            pred_score += e['cplex']
        else:
            raise Exception(pred)
        chuffed_score += e['chuffed']
        cp_sat_score += e['cp-sat']
        cplex_score += e['cplex']
        vbs_score += max(e['chuffed'], e['cp-sat'], e['cplex'])

    print(f"accuracy: {accuracy:.3f}")
    print('scores:', pred_score, chuffed_score, cp_sat_score, cplex_score, vbs_score)
    print(f"predicted score as a percentage of the virtual best: {vbs_score/pred_score:.3f}")
    print(f"cuffed score as a percentage of the virtual best: {vbs_score/chuffed_score:.3f}")
    print(f"cp-sat score as a percentage of the virtual best: {vbs_score/cp_sat_score:.3f}")
    print(f"cplex score as a percentage of the virtual best: {vbs_score/cplex_score:.3f}")
    print(f"predicted score as a percentage of the chuffed score: {pred_score/chuffed_score:.3f}")
    print(f"predicted score as a percentage of the cp-sat score: {pred_score/cp_sat_score:.3f}")
    print(f"predicted score as a percentage of the cplex score: {pred_score/cplex_score:.3f}")

    return {
        'accuracy': float(accuracy),
        'clf_score': float(pred_score),
        'vbs_score': float(vbs_score),
        'chuffed_score': float(chuffed_score),
        'cp-sat_score': float(cp_sat_score),
        'cplex_score': float(cplex_score),
        'clf_vbs': float(vbs_score/pred_score),
        'chuffed_vbs': float(vbs_score/chuffed_score),
        'cp-sat_vbs': float(vbs_score/cp_sat_score),
        'clf_chuffed': float(pred_score/chuffed_score),
        'clf_cp-sat': float(pred_score/cp_sat_score),
        'predictions': predictions,
        'hyperparameters': hyperparam
        }

def train_and_test_nn_torch(train_data:list[dict], test_data:list[dict], is_wl:bool=True, is_wlc:bool=True, use_gpu:bool=False) -> dict:
    mp.set_start_method('spawn', force=True)

    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    scores = np.array([[e['cp-sat'], e['chuffed'], e['cplex']] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    hyperparam = find_hyperparameters_nn_torch(X_train, y_train, scores, 1, use_gpu=use_gpu)
    features = X_train.shape[1]
    layer_sizes = (features, features * 2, features, features // 2)

    device = 'cuda' if use_gpu else 'auto'
    clf = TorchMLPWrapper(**hyperparam, hidden_layer_sizes=layer_sizes, device=device)
    print('hyperparameters:', hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, scores, cv=3)))

    clf.fit(X_train, y_train)
    return test_nn_torch(clf, X_test, y_test, test_data, hyperparam)
