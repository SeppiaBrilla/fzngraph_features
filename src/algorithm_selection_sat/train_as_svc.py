import numpy as np
from sklearn.svm import SVC
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

N_COMPONENTS = 100


def prepare_cv_folds(
    X: np.ndarray,
    y: np.ndarray,
    par_times: np.ndarray,
    cv: int = 5,
    n_components: int = N_COMPONENTS,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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

    folds_data = []
    for fold, (train_idx, val_idx) in enumerate(split_gen):
        X_train_fold, X_val_fold = X[train_idx].copy(), X[val_idx].copy()
        y_train_fold = y[train_idx]
        par_val_fold = par_times[val_idx]

        if X.shape[1] > n_components:
            pca = PCA(n_components, random_state=42)
            X_train_fold = pca.fit_transform(X_train_fold)
            X_val_fold = pca.transform(X_val_fold)

        folds_data.append((X_train_fold, y_train_fold, X_val_fold, par_val_fold))

    return folds_data


def evaluate_folds_svc(params: dict, folds_data: list) -> float:
    pred_pars = []
    for X_train_fold, y_train_fold, X_val_fold, par_val in folds_data:
        clf = SVC(**params)
        clf.fit(X_train_fold, y_train_fold)
        pred = clf.predict(X_val_fold)

        vbs_time = np.min(par_val, axis=1).sum()
        sbs_time = np.min(np.sum(par_val, axis=0))

        fold_par = np.sum([par_val[i, p] for i, p in enumerate(pred)])
        denom = sbs_time - vbs_time
        pred_pars.append((fold_par - vbs_time) / denom if denom > 0 else 0.0)

    return float(np.mean(pred_pars))


def cross_val_score(clf: Pipeline | SVC, X: np.ndarray, y: np.ndarray, par_times: np.ndarray, cv: int = 5) -> float:
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

    pred_pars = []
    for fold, (train_idx, val_idx) in enumerate(split_gen):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        par_val = par_times[val_idx]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)

        vbs_time = np.min(par_val, axis=1).sum()
        sbs_time = np.min(np.sum(par_val, axis=0))

        fold_par = np.sum([par_val[i, p] for i, p in enumerate(pred)])
        denom = sbs_time - vbs_time
        pred_pars.append((fold_par - vbs_time) / denom if denom > 0 else 0.0)

    return float(np.mean(pred_pars))


def _evaluate_combination(params: dict, folds_data: list) -> dict:
    np.random.seed(42)
    random.seed(42)
    par_score = evaluate_folds_svc(params, folds_data)
    return {"params": params, "score": par_score}


def find_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    par_times: np.ndarray,
    n_jobs: int = 1,
) -> dict:
    param_grid = {
        'C': np.logspace(-1, 1, 4),
        'kernel': ['rbf', 'poly', 'linear'],
        'gamma': np.logspace(-1, 1, 2).tolist() + ['scale', 'auto'],
        'shrinking': [True, False],
        'probability': [True, False],
        'max_iter': [50000],
        'random_state': [42]
    }
    all_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(all_combinations)

    folds_data = prepare_cv_folds(X, y, par_times, cv=5)
    worker_fn = partial(_evaluate_combination, folds_data=folds_data)

    results = []
    with Pool(processes=n_jobs) as pool:
        with tqdm(
            total=n_combinations,
            desc="hyperparameter search (SVC)",
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
            x['param']['C'],
        )
    )
    print('best config:', best_config)
    return best_config['param']


def test_svc(
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


def train_and_test_svc(
    train_data: list[dict],
    test_data: list[dict],
    is_wl: bool = True,
    hyperparam: None | dict = None,
    use_pca: bool = True,
) -> dict:
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    solvers = train_data[0]['solvers']
    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    par_train = np.array([[e[s] for s in solvers] for e in train_data])

    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])
    par_test = np.array([[e[s] for s in solvers] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    N_COMPONENTS_VAL = 100 if use_pca else 100000
    if hyperparam is None:
        hyperparam = find_hyperparameters(X_train, y_train, par_train, 10)

    if len(train_data[0]['features']) > N_COMPONENTS_VAL:
        clf = Pipeline([
            ('pca', PCA(N_COMPONENTS_VAL, random_state=42)),
            ('svc', SVC(**hyperparam))
        ])
    else:
        clf = Pipeline([
            ('svc', SVC(**hyperparam))
        ])

    print('hyperparameters:', hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, par_train, cv=3)))

    clf.fit(X_train, y_train)
    return test_svc(clf, X_test, y_test, par_test, par_train, solvers, test_data, hyperparam)
