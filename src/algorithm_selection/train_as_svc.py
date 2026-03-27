import numpy as np
from sklearn.svm import SVC
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.decomposition import PCA
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random, math
import multiprocessing as mp

def cross_val_score(clf:SVC, X:np.ndarray, y:np.ndarray, times:np.ndarray, cv:int=5) -> float:
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        times_val = times[val_idx]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        pred_time = sum([times_val[i,p] for i,p in enumerate(pred)])
        t0 = sum([times_val[i,0] for i,_ in enumerate(pred)])
        t1 = sum([times_val[i,1] for i,_ in enumerate(pred)])
        sb_time = min(t1, t0)

        scores.append(pred_time/sb_time)

    return float(np.mean(scores))

def _evaluate_combination(params: dict, X: np.ndarray, y: np.ndarray, times:np.ndarray) -> dict:
    np.random.seed(42)
    random.seed(42)
    model = SVC(**params, class_weight={0: 1, 1: 1, 2: 1})
    score = cross_val_score(model, X, y, times, cv=5)
    return {"params": params, "score": score}

def find_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    times:np.ndarray,
    n_jobs: int,
    ) -> dict:

    #parameters: https://readmedium.com/support-vector-machine-svm-hyperparameter-tuning-in-python-a65586289bcb
    param_grid = {
        'C': np.logspace(-1, 1, 5),
        'kernel': ['rbf', 'poly'],
        'gamma':  ['scale', 'auto'],
        'shrinking': [True, False],
        'probability': [True, False],
        'max_iter': [15000],
        'random_state': [42]
    }
    all_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(all_combinations)

    n_workers = n_jobs

    worker_fn = partial(_evaluate_combination, X=X, y=y, times=times)

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
 

    best_score = min(rows, key=lambda x: x['score'])['score']
    equivalent_scores = [r for r in rows if math.isclose(r['score'], best_score, rel_tol=0.1)]
    best_config = min(equivalent_scores, key=lambda x: (x['param']['C'],
                                                        x['param']['gamma'] if isinstance(x['param']['gamma'], (int,float)) else 0,
                                                        0 if x['param']['kernel'] == 'rbf' else 1))
    print('best config:', best_config)
 
    return best_config['param']

def size_evaluate(param:dict, hyperparams:dict, X:np.ndarray, y:np.ndarray, times:np.ndarray) -> tuple[int|None,float]:
    np.random.seed(42)
    random.seed(42)
    size = param['feature_size']
    clf = SVC(**hyperparams, class_weight={0: 1, 1: 1, 2: 1})
    if size is not None:
        pca = PCA(size, random_state=42)
        X_small = pca.fit_transform(X)
        X_small = MinMaxScaler().fit_transform(X_small)
    else:
        X_small = MinMaxScaler().fit_transform(X)

    score = cross_val_score(clf, X_small, y, times, 3)

    return size, score

def find_size(X:np.ndarray, y:np.ndarray, times:np.ndarray, hyperparams:dict, is_wl:bool) -> int|None:

    param_grid = {
        'feature_size': [n for n in range(20, min(X.shape) + 1, 20)] + [None],
    }

    all_combinations = list(ParameterGrid(param_grid))

    results = []
    for comb in tqdm(all_combinations):
        results.append(size_evaluate(comb, hyperparams, X, y, times))
 
    rows = []
    for r in sorted(results, key=lambda x:x[0] if x[0] else 1000):
        row = r
        rows.append(row)
 
    best_config = min(rows, key=lambda x: x[1])
    print('best config:', best_config)
 
    return best_config[0]

def test_svc(clf:SVC, X_test:np.ndarray, y_test:np.ndarray, test_data:list[dict], hyperparam:dict) -> dict:
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    pred_time = 0
    chuffed_time = 0
    cp_sat_time_time = 0
    vbs_time = 0
    for i, e in enumerate(test_data):
        x = np.array([X_test[i]])
        pred = clf.predict(x)[0]
        if pred == 0 or pred == 2:
            pred_time += e['chuffed']
        elif pred == 1:
            pred_time += e['cp-sat']
        else:
            raise Exception(pred)
        chuffed_time += e['chuffed']
        cp_sat_time_time += e['cp-sat']
        vbs_time += min(e['chuffed'], e['cp-sat'])

    print(f"accuracy: {accuracy:.3f}")
    print(f"predicted time as a percentage of the virtual best: {pred_time/vbs_time:.3f}")
    print(f"cuffed time as a percentage of the virtual best: {chuffed_time/vbs_time:.3f}")
    print(f"cp-sat time as a percentage of the virtual best: {cp_sat_time_time/vbs_time:.3f}")
    print(f"predicted time as a percentage of the chuffed time: {pred_time/chuffed_time:.3f}")
    print(f"predicted time as a percentage of the cp-sat time: {pred_time/cp_sat_time_time:.3f}")

    return {
        'accuracy': float(accuracy),
        'clf_time': float(pred_time),
        'vbs_time': float(vbs_time),
        'chuffed_time': float(chuffed_time),
        'cp-sat_time': float(cp_sat_time_time),
        'clf_vbs': float(pred_time/vbs_time),
        'chuffed_vbs': float(chuffed_time/vbs_time),
        'cp-sat_vbs': float(cp_sat_time_time/vbs_time),
        'clf_chuffed': float(pred_time/chuffed_time),
        'clf_cp-sat': float(pred_time/cp_sat_time_time),
        'hyperparameters': hyperparam
        }

def train_and_test_svc(train_data:list[dict], test_data:list[dict], is_wl:bool=True, is_wlc:bool=True) -> dict:
    mp.set_start_method('spawn', force=True)

    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    times = np.array([[e['chuffed'], e['cp-sat'], e['cp-sat']] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    magnitudes = np.sum(X_train, axis=0)
    sorted_indices = np.argsort(magnitudes)

    X_train = X_train[:, sorted_indices]
    X_test  = X_test[:, sorted_indices]

    hyperparam = find_hyperparameters(X_train, y_train, times, 10)
    size = find_size(X_train, y_train, times, hyperparam, is_wl)
    if not size is None:
        pca = PCA(n_components=size, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        min_max = MinMaxScaler()
        X_train = min_max.fit_transform(X_train)
        X_test = min_max.transform(X_test)


    clf = SVC(**hyperparam, class_weight={0: 1, 1: 1, 2: 1})
    print('hyperparameters:', hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, times, cv=5)))
    hyperparam['size'] = size

    clf.fit(X_train, y_train)
    test_svc(clf, X_train, y_train, train_data, hyperparam)
    return test_svc(clf, X_test, y_test, test_data, hyperparam)
