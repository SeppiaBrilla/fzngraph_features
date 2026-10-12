import numpy as np
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
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.torch_mlp import TorchMLPWrapper

LAYER_SIZE = (150, 200, 150)
N_COMPONENTS = 100


def cross_val_score(clf: Pipeline, X: np.ndarray, y: np.ndarray, par_times: np.ndarray, cv: int = 5) -> float:
    quantiles = np.linspace(0, 100, 8)
    gap = np.abs(par_times.max(axis=1) - par_times.min(axis=1))
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
        par_val = par_times[val_idx]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        vbs_time = np.min(par_val, axis=1).sum()
        sbs_time = np.min(np.sum(par_val, axis=0))

        fold_par = np.sum([par_val[i, p] for i, p in enumerate(pred)])
        denom = sbs_time - vbs_time
        fold_scores.append((fold_par - vbs_time) / denom if denom > 0 else 0.0)

    return float(np.mean(fold_scores))


def _evaluate_combination(params: dict, X: np.ndarray, y: np.ndarray, par_times: np.ndarray, num_classes: int) -> dict:
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    device = 'auto'
    steps = []
    if X.shape[1] > N_COMPONENTS:
        steps.append(('pca', PCA(n_components=N_COMPONENTS, random_state=42)))
    steps.append(('torch', TorchMLPWrapper(**params, hidden_layer_sizes=LAYER_SIZE, device=device, num_classes=num_classes)))
    model = Pipeline(steps)
    score = cross_val_score(model, X, y, par_times, cv=3)
    return {"params": params, "score": score}


def find_hyperparameters_nn_torch(
    X: np.ndarray,
    y: np.ndarray,
    par_times: np.ndarray,
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

    worker_fn = partial(_evaluate_combination, X=X, y=y, par_times=par_times, num_classes=num_classes)

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

    best_score = min(rows, key=lambda x: x['score'])['score']
    equivalent_scores = [r for r in rows if math.isclose(r['score'], best_score, rel_tol=0.001)]
    best_config = min(
        equivalent_scores,
        key=lambda x: (
            x['param']['alpha'],
        )
    )
    print('best config:', best_config)
    return best_config['param']


def test_torch_neural_network(
    clf: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    par_test: np.ndarray,
    par_train: np.ndarray,
    solvers: list[str],
    test_data: list[dict],
    hyperparam: dict
) -> dict:
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    clf_par_vals = [par_test[i, p] for i, p in enumerate(pred)]
    clf_par = float(np.mean(clf_par_vals))
    vbs_par = float(np.mean(np.min(par_test, axis=1)))

    sbs_idx = int(np.argmin(np.mean(par_train, axis=0)))
    sbs_solver_name = solvers[sbs_idx]
    sbs_par = float(np.mean(par_test[:, sbs_idx]))
    sbs_test_par = float(np.min(np.mean(par_test, axis=0)))

    solvers_par = {s: float(np.mean(par_test[:, idx])) for idx, s in enumerate(solvers)}

    denom = sbs_par - vbs_par
    score = float((clf_par - vbs_par) / denom) if denom > 0 else 0.0

    predictions = {}
    for i, e in enumerate(test_data):
        inst_key = e.get('flatzinc', f"{e.get('name', i)}")
        predictions[inst_key] = {
            'pred': int(pred[i]),
            'true': int(y_test[i]),
            'pred_solver': solvers[pred[i]],
            'true_solver': solvers[y_test[i]],
            'runtime': float(par_test[i, pred[i]]),
        }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Selector Runtime/PAR2: {clf_par:.2f} | SBS ({sbs_solver_name}): {sbs_par:.2f} | VBS: {vbs_par:.2f}")
    print(f"Score (pred - vbs) / (sbs - vbs): {score:.4f} | Headroom: {1.0 - score:.4f}")

    return {
        'accuracy': float(accuracy),
        'score': float(score),
        'clf_score': float(score),
        'clf_par10': float(clf_par),
        'vbs_par10': float(vbs_par),
        'sbs_par10': float(sbs_par),
        'sbs_test_par10': float(sbs_test_par),
        'sbs_solver': str(sbs_solver_name),
        'solvers_par10': solvers_par,
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
    par_train = np.array([[e[s] for s in solvers] for e in train_data])

    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])
    par_test = np.array([[e[s] for s in solvers] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    N_COMPONENTS_VAL = 100 if use_pca else X_train.shape[1]

    if hyperparams is None:
        hyperparam = find_hyperparameters_nn_torch(X_train, y_train, par_train, num_classes, 1)
    else:
        hyperparam = hyperparams

    device = 'auto'
    steps = []
    if len(train_data[0]['features']) > N_COMPONENTS_VAL:
        steps.append(('pca', PCA(n_components=N_COMPONENTS_VAL, random_state=42)))
    steps.append(('torch', TorchMLPWrapper(**hyperparam, hidden_layer_sizes=LAYER_SIZE, device=device, num_classes=num_classes)))
    clf = Pipeline(steps)
    print('hyperparameters:', hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, par_train, cv=3)))

    clf.fit(X_train, y_train)
    return test_torch_neural_network(clf, X_test, y_test, par_test, par_train, solvers, test_data, hyperparam)
