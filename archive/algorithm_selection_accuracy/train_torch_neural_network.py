import numpy as np
from common.torch_mlp import TorchMLPWrapper
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.decomposition import PCA
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random, math
import multiprocessing as mp
import os

os.environ["OMP_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"

def cross_val_score(clf: TorchMLPWrapper, X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    pred_scores = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        pred_scores.append(accuracy_score(y_val, pred))

    return float(np.mean(pred_scores))

def _evaluate_combination(params: dict, X: np.ndarray, y: np.ndarray) -> dict:
    np.random.seed(42)
    random.seed(42)

    layer_sizes = (100, 100, 50)

    device = 'auto'
    model = TorchMLPWrapper(**params, hidden_layer_sizes=layer_sizes, device=device)
    score = cross_val_score(model, X, y, cv=3)
    return {"params": params, "score": score}

def find_hyperparameters_nn(
    X: np.ndarray,
    y: np.ndarray,
    n_jobs: int,
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

    worker_fn = partial(_evaluate_combination, X=X, y=y)

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
        row = {'param': r["params"], "score": r["score"]}
        rows.append(row)

    best_score = max(rows, key=lambda x: x['score'])['score']
    equivalent_scores = [r for r in rows if math.isclose(r['score'], best_score, rel_tol=0.001)]
    best_config = min(equivalent_scores, key=lambda x: (x['param']['max_iter'], x['param']['alpha']))
    print('best config:', best_config)

    return best_config['param']

def test_nn(clf: TorchMLPWrapper, X_test: np.ndarray, y_test: np.ndarray, test_data: list[dict], hyperparam: dict) -> dict:
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    predictions = {}

    for i, e in enumerate(test_data):
        x = np.array([X_test[i]])
        pred = clf.predict(x)[0]
        predictions[f"{e['model']}-sep-{e['name']}"] = {'pred': int(pred), 'true': int(e['label'])}

    print(f"accuracy: {accuracy:.3f}")
    return {
        'accuracy': float(accuracy),
        'hyperparameters': hyperparam,
        'predictions': predictions
    }

def train_and_test_nn_torch(train_data: list[dict], test_data: list[dict], hyperparams:None|dict=None) -> dict:
    mp.set_start_method('spawn', force=True)

    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if hyperparams is None:
        hyperparam = find_hyperparameters_nn(X_train, y_train, 1)
    else:
        hyperparam = hyperparams
    layer_sizes = (100, 100, 50)

    device = 'auto'
    clf = TorchMLPWrapper(**hyperparam, hidden_layer_sizes=layer_sizes, device=device)
    print('hyperparameters:', hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, cv=3)))

    clf.fit(X_train, y_train)
    return test_nn(clf, X_test, y_test, test_data, hyperparam)
