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


def cross_val_score(clf: Pipeline, X: np.ndarray, y: np.ndarray, scores: np.ndarray, cv: int = 5) -> float:
    gap = scores.max(axis=1) - scores.min(axis=1)
    quantiles = np.linspace(0, 100, 8)
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
        X_train_fold, X_val_fold = X[train_idx].copy(), X[val_idx].copy()
        y_train_fold = y[train_idx]
        scores_val_fold = scores[val_idx]

        n_c = min(X_train_fold.shape[0], X_train_fold.shape[1], N_COMPONENTS)
        steps = []
        if X_train_fold.shape[1] > n_c:
            steps.append(('pca', PCA(n_components=n_c, random_state=42)))
        torch_step = clf.named_steps['torch'] if isinstance(clf, Pipeline) else clf
        steps.append(('torch', torch_step))
        fold_clf = Pipeline(steps)

        fold_clf.fit(X_train_fold, y_train_fold)
        pred = fold_clf.predict(X_val_fold)

        pred_score = sum([scores_val_fold[i, p] for i, p in enumerate(pred)])
        sb_score = max([sum(scores_val_fold[:, s_idx]) for s_idx in range(scores_val_fold.shape[1])])
        fold_scores.append(pred_score / sb_score if sb_score > 0 else 0.0)

    return float(np.mean(fold_scores))


def _evaluate_combination(params: dict, X: np.ndarray, y: np.ndarray, scores: np.ndarray, num_classes: int) -> dict:
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    device = 'auto'
    steps = []
    n_c = min(X.shape[0], X.shape[1], N_COMPONENTS)
    if X.shape[1] > n_c:
        steps.append(('pca', PCA(n_components=n_c, random_state=42)))
    steps.append(('torch', TorchMLPWrapper(**params, hidden_layer_sizes=LAYER_SIZE, device=device, num_classes=num_classes)))
    model = Pipeline(steps)
    score = cross_val_score(model, X, y, scores, cv=3)
    return {"params": params, "score": score}


def find_hyperparameters_nn_torch(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
    n_jobs: int = 1,
) -> dict:
    param_grid = {
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant'],
        'max_iter': [15000],
        'random_state': [42]
    }
    all_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(all_combinations)

    worker_fn = partial(_evaluate_combination, X=X, y=y, scores=scores, num_classes=num_classes)

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

    best_score = max(rows, key=lambda x: x['score'])['score']
    equivalent_scores = [r for r in rows if math.isclose(r['score'], best_score, rel_tol=0.001)]
    best_config = min(equivalent_scores, key=lambda x: (x['param']['max_iter'], x['param']['alpha']))
    print('best config:', best_config)

    return best_config['param']


def test_torch_neural_network(
    clf: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scores_test: np.ndarray,
    scores_train: np.ndarray,
    solvers: list[str],
    test_data: list[dict],
    hyperparam: dict
) -> dict:
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    clf_score = float(sum([scores_test[i, p] for i, p in enumerate(pred)]))
    vbs_score = float(scores_test.max(axis=1).sum())

    sbs_idx = int(np.argmax(scores_train.sum(axis=0)))
    sbs_solver_name = solvers[sbs_idx]
    sbs_score = float(scores_test[:, sbs_idx].sum())

    denom = vbs_score - sbs_score
    norm_score = float((clf_score - sbs_score) / denom) if denom > 1e-6 else 0.0

    solver_scores = {s: float(scores_test[:, idx].sum()) for idx, s in enumerate(solvers)}

    predictions = {}
    for i, e in enumerate(test_data):
        inst_key = f"{e.get('model', '')}-sep-{e.get('name', i)}"
        predictions[inst_key] = {
            'pred': int(pred[i]),
            'true': int(y_test[i]),
            'pred_solver': solvers[pred[i]],
            'true_solver': solvers[y_test[i]],
            'score': float(scores_test[i, pred[i]]),
        }

    print(f"NN Accuracy: {accuracy:.4f}")
    print(f"NN Selector Score: {clf_score:.2f} | SBS ({sbs_solver_name}): {sbs_score:.2f} | VBS: {vbs_score:.2f}")
    print(f"Norm Score (clf - sbs) / (vbs - sbs): {norm_score:.4f}")

    return {
        'accuracy': float(accuracy),
        'score': float(norm_score),
        'clf_score': float(clf_score),
        'vbs_score': float(vbs_score),
        'sbs_score': float(sbs_score),
        'sbs_solver': str(sbs_solver_name),
        'norm_score': float(norm_score),
        'solver_scores': solver_scores,
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
    scores_train = np.array([[e[s] for s in solvers] for e in train_data])

    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])
    scores_test = np.array([[e[s] for s in solvers] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_c = min(X_train.shape[0], X_train.shape[1], N_COMPONENTS) if use_pca else X_train.shape[1]

    if hyperparams is None:
        hyperparam = find_hyperparameters_nn_torch(X_train, y_train, scores_train, num_classes, 1)
    else:
        hyperparam = hyperparams

    device = 'auto'
    steps = []
    if X_train.shape[1] > n_c:
        steps.append(('pca', PCA(n_components=n_c, random_state=42)))
    steps.append(('torch', TorchMLPWrapper(**hyperparam, hidden_layer_sizes=LAYER_SIZE, device=device, num_classes=num_classes)))
    clf = Pipeline(steps)

    print('hyperparameters:', hyperparam)
    print("Inner CV score:", np.mean(cross_val_score(clf, X_train, y_train, scores_train, cv=3)))

    clf.fit(X_train, y_train)
    return test_torch_neural_network(clf, X_test, y_test, scores_test, scores_train, solvers, test_data, hyperparam)
