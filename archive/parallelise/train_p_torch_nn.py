import numpy as np
from common.torch_mlp import TorchMLPWrapper
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, cross_val_score
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random, math
import multiprocessing as mp

def _evaluate_combination(params: dict, X: np.ndarray, y: np.ndarray, rnd_state: int, use_gpu: bool = False) -> dict:
    np.random.seed(42)
    random.seed(42)

    features = X.shape[1]
    layer_sizes = (features, features * 2, features, features // 2)

    device = 'cuda' if use_gpu else 'auto'
    model = TorchMLPWrapper(**params, hidden_layer_sizes=layer_sizes, random_state=rnd_state, device=device)
    cv = 3
    scores = cross_val_score(model, X, y, cv=cv)
    return {"params": params, "score": np.mean(scores)}

def find_hyperparameters_nn(
    train_data: list[dict],
    rnd_state: int,
    n_jobs: int,
    use_gpu: bool = False,
) -> dict:

    param_grid = {
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [15000],
    }

    all_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(all_combinations)

    X = np.array([e['features'] for e in train_data])
    y = np.array([e['label'] for e in train_data])

    X = MinMaxScaler().fit_transform(X)

    n_workers = n_jobs

    worker_fn = partial(_evaluate_combination, X=X, y=y, rnd_state=rnd_state, use_gpu=use_gpu)

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
    equivalent_scores = [r for r in rows if math.isclose(r['score'], best_score, rel_tol=0.1)]
    best_config = min(equivalent_scores, key=lambda x: (x['param']['alpha']))
    print('best config:', best_config)

    return best_config['param']

def test_nn(clf: TorchMLPWrapper, X_test: np.ndarray, y_test: np.ndarray, test_data: list[dict], hyperparam: dict) -> dict:
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    predictions = {}

    for i, e in enumerate(test_data):
        x = np.array([X_test[i]])
        pred = int(clf.predict(x)[0])
        predictions[f"{e['model']}-sep-{e['name']}"] = {'pred': pred, 'true': int(e['label'])}

    print(f"accuracy: {accuracy:.3f}")
    return {
        'accuracy': float(accuracy),
        'hyperparameters': hyperparam,
        'predictions': predictions
    }

def train_and_test_nn_torch(train_data: list[dict], test_data: list[dict], rnd_state: int, use_gpu: bool = False) -> dict:
    mp.set_start_method('spawn', force=True)

    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    hyperparam = find_hyperparameters_nn(train_data, rnd_state, 1, use_gpu=use_gpu)

    features = X_train.shape[1]
    layer_sizes = (features, features * 2, features, features // 2)

    device = 'cuda' if use_gpu else 'auto'
    clf = TorchMLPWrapper(**hyperparam, hidden_layer_sizes=layer_sizes, random_state=rnd_state, device=device)
    print('hyperparameters:', hyperparam)

    clf.fit(X_train, y_train)
    return test_nn(clf, X_test, y_test, test_data, hyperparam)
