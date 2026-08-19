import numpy as np
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.decomposition import PCA
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random, math
import multiprocessing as mp

def cross_val_score(clf:RandomForestClassifier, X:np.ndarray, y:np.ndarray, cv:int=5) -> float:
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
    model = RandomForestClassifier(**params, class_weight={0: 1, 1: 1, 2: 1})
    score = cross_val_score(model, X, y, cv=3)
    return {"params": params, "score": score}

def find_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_jobs: int,
    ) -> dict:

    #parameters: https://www.researchgate.net/figure/Tested-parameter-grid-for-random-forest-classifier_tbl1_350998771
    param_grid = {
        'n_estimators': [n for n in range(200, 1001, 200)],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [n for n in range(10, 101, 10)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
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
        row = {'param':r["params"], "score": r["score"]}
        rows.append(row)
 
    best_score = max(rows, key=lambda x: x['score'])['score']
    equivalent_scores = [r for r in rows if math.isclose(r['score'], best_score, rel_tol=0.001)]
    best_config = min(equivalent_scores, key=lambda x: (x['param']['n_estimators'], x['param']['max_depth'] if x['param']['max_depth'] else 1000))
    print('best config:', best_config)
 
    return best_config['param']

def size_evaluate(param:dict, hyperparams:dict, X:np.ndarray, y:np.ndarray) -> tuple[int|None,float]:
    np.random.seed(42)
    random.seed(42)
    size = param['feature_size']
    clf = RandomForestClassifier(**hyperparams, class_weight={0: 1, 1: 1, 2: 1}, n_jobs=1)
    if size is not None:
        pca = PCA(size, random_state=42)
        X_small = pca.fit_transform(X)
    else:
        X_small = X

    score = cross_val_score(clf, X_small, y, 3)

    return size, score

def find_size(X:np.ndarray, y:np.ndarray, hyperparams:dict, is_wl:bool) -> int|None:

    param_grid = {
        'feature_size': [n for n in range(20, min(X.shape) + 1, 20)] + [None],
    }

    all_combinations = list(ParameterGrid(param_grid))

    results = []
    for comb in tqdm(all_combinations):
        results.append(size_evaluate(comb, hyperparams, X, y))
 
    rows = []
    for r in sorted(results, key=lambda x:x[0] if x[0] else 1000):
        row = r
        rows.append(row)
 
    best_config = max(rows, key=lambda x: x[1])
    print('best config:', best_config)
 
    return best_config[0]

def test_rnd_forest(clf:RandomForestClassifier, X_test:np.ndarray, y_test:np.ndarray, test_data:list[dict], hyperparam:dict) -> dict:
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    predictions = {}

    for i, e in enumerate(test_data):
        x = np.array([X_test[i]])
        pred = int(clf.predict(x)[0])
        predictions[f"{e['model']}-sep-{e['name']}"] = {'pred':int(pred), 'true':int(e['label'])}

    print(f"accuracy: {accuracy:.3f}")

    return {
        'accuracy': float(accuracy),
        'hyperparameters': hyperparam,
        'predictions':predictions
        }

def train_and_test_rnd_forest(train_data:list[dict], test_data:list[dict], is_wl:bool=True, is_wlc:bool=True, hyperparams:None|dict=None) -> dict:
    mp.set_start_method('spawn', force=True)

    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    if is_wlc:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        magnitudes = np.sum(X_train, axis=0)
        sorted_indices = np.argsort(magnitudes)

        X_train = X_train[:, sorted_indices]
        X_test  = X_test[:, sorted_indices]

    if hyperparams is None:
        hyperparam = find_hyperparameters(X_train, y_train, 5)
    else:
        hyperparam = hyperparams
    # size = find_size(X_train, y_train, hyperparam, is_wl)
    # if not size is None:
        # pca = PCA(n_components=size, random_state=42)
        # X_train = pca.fit_transform(X_train)
        # X_test = pca.transform(X_test)

    clf = RandomForestClassifier(**hyperparam, class_weight={0: 1, 1: 1, 2: 1})
    print('hyperparameters:', hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, cv=3)))
    # hyperparam['size'] = size

    clf.fit(X_train, y_train)
    return test_rnd_forest(clf, X_test, y_test, test_data, hyperparam)
