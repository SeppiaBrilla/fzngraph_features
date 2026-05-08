import numpy as np
from sklearn.svm import SVC
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.decomposition import PCA
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random, math
import multiprocessing as mp

def cross_val_score(clf:SVC, X:np.ndarray, y:np.ndarray, scores:np.ndarray, cv:int=5) -> float:
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    pred_scores = []
    quantiles = np.linspace(0, 100, 8)
    gap = np.abs(scores.max(axis=1) - scores.min(axis=1))
    bins = np.unique(np.percentile(gap, quantiles))
    buckets = np.digitize(gap, bins[1:-1])
    for train_idx, val_idx in kf.split(X, buckets):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        scores_val = scores[val_idx]
        # weights = 1 + gap[train_idx] / np.max(gap[train_idx])

        clf.fit(X_train, y_train)#, sample_weight=weights)
        pred = clf.predict(X_val)
        pred_score = sum([scores_val[i,p] for i,p in enumerate(pred)])
        t0 = sum([scores_val[i,0] for i,_ in enumerate(pred)])
        t1 = sum([scores_val[i,1] for i,_ in enumerate(pred)])
        t2 = sum([scores_val[i,2] for i,_ in enumerate(pred)])
        sb_score = max(t1, t0, t2)

        pred_scores.append(pred_score/sb_score)

    return float(np.mean(pred_scores))

def _evaluate_combination(params: dict, X: np.ndarray, y: np.ndarray, scores:np.ndarray) -> dict:
    np.random.seed(42)
    random.seed(42)
    model = SVC(**params, class_weight={0: 1, 1: 1, 2: 1})
    score = cross_val_score(model, X, y, scores, cv=5)
    return {"params": params, "score": score}

def find_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    scores:np.ndarray,
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

    worker_fn = partial(_evaluate_combination, X=X, y=y, scores=scores)

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
    best_config = min(equivalent_scores, key=lambda x: (x['param']['C'],
                                                        x['param']['gamma'] if isinstance(x['param']['gamma'], (int,float)) else 0,
                                                        0 if x['param']['kernel'] == 'rbf' else 1))
    print('best config:', best_config)
 
    return best_config['param']

def size_evaluate(param:dict, hyperparams:dict, X:np.ndarray, y:np.ndarray, scores:np.ndarray) -> tuple[int|None,float]:
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

    score = cross_val_score(clf, X_small, y, scores, 3)

    return size, score

def find_size(X:np.ndarray, y:np.ndarray, scores:np.ndarray, hyperparams:dict, is_wl:bool) -> int|None:

    param_grid = {
        'feature_size': [n for n in range(20, min(X.shape) + 1, 20)] + [None],
    }

    all_combinations = list(ParameterGrid(param_grid))

    results = []
    for comb in tqdm(all_combinations):
        results.append(size_evaluate(comb, hyperparams, X, y, scores))
 
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

def train_and_test_svc(train_data:list[dict], test_data:list[dict], is_wl:bool=True, is_wlc:bool=True) -> dict:
    mp.set_start_method('spawn', force=True)

    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    # times = np.array([[e['chuffed'], e['cp-sat'], e['cp-sat']] for e in train_data])
    scores = np.array([[e['cp-sat'], e['chuffed'], e['cplex']] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    magnitudes = np.sum(X_train, axis=0)
    sorted_indices = np.argsort(magnitudes)

    X_train = X_train[:, sorted_indices]
    X_test  = X_test[:, sorted_indices]

    hyperparam = find_hyperparameters(X_train, y_train, scores, 5)
    size = find_size(X_train, y_train, scores, hyperparam, is_wl)
    # size = None
    if not size is None:
        pca = PCA(n_components=size, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        min_max = MinMaxScaler()
        X_train = min_max.fit_transform(X_train)
        X_test = min_max.transform(X_test)


    clf = SVC(**hyperparam, class_weight={0: 1, 1: 1, 2: 1})
    print('hyperparameters:', hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, scores, cv=5)))
    hyperparam['size'] = size

    # weights = np.abs(times[:, 0] - times[:, 1])
    # weights = 1 + weights / np.max(weights)

    clf.fit(X_train, y_train)#, sample_weight=weights)
    test_svc(clf, X_train, y_train, train_data, hyperparam)
    return test_svc(clf, X_test, y_test, test_data, hyperparam)
