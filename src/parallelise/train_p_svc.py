from copy import deepcopy
import numpy as np
from sklearn.svm import SVC
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.decomposition import PCA
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random, math
import multiprocessing as mp

def _evaluate_combination(params: dict, X:np.ndarray, y:np.ndarray) -> dict:
    model = SVC(**params)
    scores = []
    cv=3
    scores = cross_val_score(model, X, y, cv=cv)
    return {"params": params, "score": np.mean(scores)}

def find_hyperparameters(
    train_data:list[dict],
    n_jobs: int,
    ) -> dict:

    #parameters: https://readmedium.com/support-vector-machine-svm-hyperparameter-tuning-in-python-a65586289bcb
    param_grid = {
        'C': np.logspace(-1, 1, 4),
        'kernel': ['rbf', 'poly', 'linear'],
        'gamma':  np.logspace(-1, 1, 2).tolist() + ['scale', 'auto'],
        'shrinking': [True, False],
        'probability': [True, False],
        'max_iter': [15000],
        'random_state': [42]
    }
    all_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(all_combinations)
 
    n_workers = n_jobs

    # scorer = CrossValidator(train_data, cv=3)
 
    X = np.array([e['features'] for e in train_data])
    y = np.array([e['label'] for e in train_data])

    X = MinMaxScaler().fit_transform(X)

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
    equivalent_scores = [r for r in rows if math.isclose(r['score'], best_score, rel_tol=0.01)]
    best_config = min(equivalent_scores, key=lambda x: (x['param']['C'],
                                                        x['param']['gamma'] if isinstance(x['param']['gamma'], (int,float)) else 0,
                                                        0 if x['param']['kernel'] == 'rbf' else 1))
    print('best config:', best_config)
 
    return best_config['param']

def size_evaluate(param:dict, hyperparams:dict, train_data:list[dict]) -> tuple[int|None,float]:
    size = param['feature_size']
    clf = SVC(**hyperparams)
    X = np.array([e['features'] for e in train_data])
    if size is not None:
        pca = PCA(size, random_state=42)
        X_small = pca.fit_transform(X)
    else:
        X_small = X
    new_train_data = deepcopy(train_data)
    for i in range(len(new_train_data)):
        new_train_data[i]['features'] = X_small[i,:]
 
    X = MinMaxScaler().fit_transform(np.array([e['features'] for e in new_train_data]))
    y = np.array([e['label'] for e in new_train_data])
    problems = [e['model'] for e in new_train_data]

    score = cross_val_score(clf, X, y)

    return size, float(np.mean(score))

def find_size(train_data:list[dict], hyperparams:dict) -> int|None:

    param_grid = {
        'feature_size': [n for n in range(20, min(len(train_data), max(train_data[0]['features'].shape)) + 1, 20)] + [None],
    }

    all_combinations = list(ParameterGrid(param_grid))

    results = []
    for comb in tqdm(all_combinations):
        results.append(size_evaluate(comb, hyperparams, train_data))
 
    rows = []
    for r in sorted(results, key=lambda x:x[0] if x[0] else 1000):
        row = r
        rows.append(row)
 
    best_config = max(rows, key=lambda x: x[1])
    print('best config:', best_config)
 
    return best_config[0]

def test_svc(clf:SVC, X_test:np.ndarray, y_test:np.ndarray, test_data:list[dict], hyperparam:dict) -> dict:
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    predictions = {}

    for i, e in enumerate(test_data):
        x = np.array([X_test[i]])
        pred = int(clf.predict(x)[0])
        predictions[f"{e['model']}-sep-{e['name']}"] = {'pred':pred, 'true':int(e['label'])}

    print(f"accuracy: {accuracy:.3f}")
    return {
        'accuracy': float(accuracy),
        'hyperparameters': hyperparam,
        'predictions': predictions
        }

def train_and_test_svc(train_data:list[dict], test_data:list[dict], random_seed:int) -> dict:
    mp.set_start_method('spawn', force=True)

    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    hyperparam = find_hyperparameters(train_data, 10)
    size = find_size(train_data, hyperparam)
    # size = 80
    if not size is None:
        pca = PCA(n_components=size, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)


    clf = SVC(**hyperparam)
    print('hyperparameters:', hyperparam)
    hyperparam['size'] = size

    clf.fit(X_train, y_train)
    return test_svc(clf, X_test, y_test, test_data, hyperparam)
