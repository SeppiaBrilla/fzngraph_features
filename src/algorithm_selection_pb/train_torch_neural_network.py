import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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

LAYER_SIZE = (150, 200, 150)
N_COMPONENTS = 100
TIMEOUT = 1800.0
PAR10_PENALTY = 10.0 * TIMEOUT


def cross_val_score(clf: Pipeline, X: np.ndarray, y: np.ndarray, par10_times: np.ndarray, cv: int = 5) -> float:
    quantiles = np.linspace(0, 100, 8)
    gap = np.abs(par10_times.max(axis=1) - par10_times.min(axis=1))
    bins = np.unique(np.percentile(gap, quantiles))
    buckets = np.digitize(gap, bins[1:-1])

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    try:
        split_gen = list(kf.split(X, buckets))
    except ValueError:
        kf_fallback = KFold(n_splits=cv, shuffle=True, random_state=42)
        split_gen = list(kf_fallback.split(X))

    fold_scores = []
    for train_idx, val_idx in split_gen:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        par10_val = par10_times[val_idx]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        vbs_time = np.min(par10_val, axis=1).sum()
        sbs_time = np.min(np.sum(par10_val, axis=0))

        fold_par10 = np.sum([par10_val[i, p] for i, p in enumerate(pred)])
        denom = sbs_time - vbs_time
        fold_scores.append((fold_par10 - vbs_time) / denom if denom > 0 else 0.0)

    return float(np.mean(fold_scores))


def _evaluate_combination(params: dict, X: np.ndarray, y: np.ndarray, par10_times: np.ndarray, num_classes: int) -> dict:
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    device = 'auto'
    steps = []
    if X.shape[1] > N_COMPONENTS:
        steps.append(('pca', PCA(n_components=N_COMPONENTS, random_state=42)))
    steps.append(('torch', TorchMLPWrapper(**params, hidden_layer_sizes=LAYER_SIZE, device=device, num_classes=num_classes)))
    model = Pipeline(steps)
    score = cross_val_score(model, X, y, par10_times, cv=3)
    return {"params": params, "score": score}


def find_hyperparameters_nn_torch(
    X: np.ndarray,
    y: np.ndarray,
    par10_times: np.ndarray,
    num_classes: int,
    n_jobs: int = 1,
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

    worker_fn = partial(_evaluate_combination, X=X, y=y, par10_times=par10_times, num_classes=num_classes)

    results = []
    with Pool(processes=n_jobs) as pool:
        with tqdm(
            total=n_combinations,
            desc="hyperparameter search (NN)",
            unit="combo",
            dynamic_ncols=True,
        ) as pbar:
            for result in pool.imap(worker_fn, all_combinations):
                results.append(result)
                pbar.set_postfix(score=f"{result['score']:.4f}", refresh=False)
                pbar.update()

    rows = []
    for r in results:
        rows.append({'param': r["params"], "score": r["score"]})

    # Minimize relative score
    best_score = min(rows, key=lambda x: x['score'])['score']
    equivalent_scores = [r for r in rows if math.isclose(r['score'], best_score, rel_tol=0.001)]
    best_config = min(equivalent_scores, key=lambda x: (x['param']['max_iter'], x['param']['alpha']))
    print('best config:', best_config)

    return best_config['param']


def test_torch_neural_network(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    par10_test: np.ndarray,
    par10_train: np.ndarray,
    solvers: list[str],
    test_data: list[dict],
    hyperparam: dict
) -> dict:
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    clf_par10_vals = [par10_test[i, p] for i, p in enumerate(pred)]
    clf_par10 = float(np.mean(clf_par10_vals))
    vbs_par10 = float(np.mean(np.min(par10_test, axis=1)))

    sbs_idx = int(np.argmin(np.mean(par10_train, axis=0)))
    sbs_solver_name = solvers[sbs_idx]
    sbs_par10 = float(np.mean(par10_test[:, sbs_idx]))
    sbs_test_par10 = float(np.min(np.mean(par10_test, axis=0)))

    solvers_par10 = {s: float(np.mean(par10_test[:, idx])) for idx, s in enumerate(solvers)}

    denom = sbs_par10 - vbs_par10
    score = float((clf_par10 - vbs_par10) / denom) if denom > 0 else 0.0

    predictions = {}
    for i, e in enumerate(test_data):
        inst_key = e.get('flatzinc', f"{e.get('model', '')}-sep-{e.get('name', i)}")
        predictions[inst_key] = {
            'pred': int(pred[i]),
            'true': int(y_test[i]),
            'pred_solver': solvers[pred[i]],
            'true_solver': solvers[y_test[i]],
            'par10': float(par10_test[i, pred[i]]),
        }

    print(f"NN Accuracy: {accuracy:.4f}")
    print(f"NN Selector PAR10: {clf_par10:.2f} | SBS ({sbs_solver_name}): {sbs_par10:.2f} | VBS: {vbs_par10:.2f}")
    print(f"Score (pred - vbs) / (sbs - vbs): {score:.4f}")

    return {
        'accuracy': float(accuracy),
        'score': float(score),
        'clf_score': float(score),
        'clf_par10': float(clf_par10),
        'vbs_par10': float(vbs_par10),
        'sbs_par10': float(sbs_par10),
        'sbs_test_par10': float(sbs_test_par10),
        'sbs_solver': str(sbs_solver_name),
        'solvers_par10': solvers_par10,
        'predictions': predictions,
        'hyperparameters': hyperparam,
    }


def train_and_test_nn_torch(
    train_data: list[dict],
    test_data: list[dict],
    hyperparams: None | dict = None,
    use_pca: bool = True,
) -> dict:
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    solvers = train_data[0]['solvers']
    num_classes = len(solvers)

    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    times_train = np.array([[e[s] for s in solvers] for e in train_data])
    par10_train = np.where(times_train < TIMEOUT, times_train, PAR10_PENALTY)

    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])
    times_test = np.array([[e[s] for s in solvers] for e in test_data])
    par10_test = np.where(times_test < TIMEOUT, times_test, PAR10_PENALTY)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    N_COMPONENTS_VAL = 100 if use_pca else X_train.shape[1]

    if hyperparams is None:
        hyperparam = find_hyperparameters_nn_torch(X_train, y_train, par10_train, num_classes, 1)
    else:
        hyperparam = hyperparams

    device = 'auto'
    steps = []
    if len(train_data[0]['features']) > N_COMPONENTS_VAL:
        steps.append(('pca', PCA(n_components=N_COMPONENTS_VAL, random_state=42)))
    steps.append(('torch', TorchMLPWrapper(**hyperparam, hidden_layer_sizes=LAYER_SIZE, device=device, num_classes=num_classes)))
    clf = Pipeline(steps)
    print('hyperparameters:', hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, par10_train, cv=3)))

    clf.fit(X_train, y_train)
    return test_torch_neural_network(clf, X_test, y_test, par10_test, par10_train, solvers, test_data, hyperparam)
