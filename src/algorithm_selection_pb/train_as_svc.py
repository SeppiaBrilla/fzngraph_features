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
TIMEOUT = 1800.0
PAR10_PENALTY = 10.0 * TIMEOUT


def prepare_cv_folds(
    X: np.ndarray,
    y: np.ndarray,
    par10_times: np.ndarray,
    cv: int = 5,
    n_components: int = N_COMPONENTS,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pre-computes and caches PCA results for each CV fold.
    This avoids re-fitting PCA repeatedly for every hyperparameter combination in parallel.
    """
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

    folds_data = []
    for fold, (train_idx, val_idx) in enumerate(split_gen):
        X_train_fold, X_val_fold = X[train_idx].copy(), X[val_idx].copy()
        y_train_fold = y[train_idx]
        par10_val_fold = par10_times[val_idx]

        if X.shape[1] > n_components:
            pca = PCA(n_components, random_state=42)
            X_train_fold = pca.fit_transform(X_train_fold)
            X_val_fold = pca.transform(X_val_fold)

        folds_data.append((X_train_fold, y_train_fold, X_val_fold, par10_val_fold))

    return folds_data


def evaluate_folds_svc(params: dict, folds_data: list) -> float:
    pred_par10s = []
    for X_train_fold, y_train_fold, X_val_fold, par10_val in folds_data:
        clf = SVC(**params)
        clf.fit(X_train_fold, y_train_fold)
        pred = clf.predict(X_val_fold)

        vbs_time = np.min(par10_val, axis=1).sum()
        sbs_time = np.min(np.sum(par10_val, axis=0))

        fold_par10 = np.sum([par10_val[i, p] for i, p in enumerate(pred)])
        denom = sbs_time - vbs_time
        pred_par10s.append((fold_par10 - vbs_time) / denom if denom > 0 else 0.0)

    return float(np.mean(pred_par10s))


def cross_val_score(clf: Pipeline | SVC, X: np.ndarray, y: np.ndarray, par10_times: np.ndarray, cv: int = 5) -> float:
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

    pred_par10s = []
    for fold, (train_idx, val_idx) in enumerate(split_gen):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        par10_val = par10_times[val_idx]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)

        vbs_time = np.min(par10_val, axis=1).sum()
        sbs_time = np.min(np.sum(par10_val, axis=0))

        fold_par10 = np.sum([par10_val[i, p] for i, p in enumerate(pred)])
        denom = sbs_time - vbs_time
        pred_par10s.append((fold_par10 - vbs_time) / denom if denom > 0 else 0.0)

    return float(np.mean(pred_par10s))


def _evaluate_combination(params: dict, folds_data: list) -> dict:
    np.random.seed(42)
    random.seed(42)
    score = evaluate_folds_svc(params, folds_data)
    return {"params": params, "score": score}


def find_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    par10_times: np.ndarray,
    n_jobs: int,
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

    # Pre-cache PCA for each CV fold ONCE to optimize parallel performance
    folds_data = prepare_cv_folds(X, y, par10_times, cv=5)

    worker_fn = partial(_evaluate_combination, folds_data=folds_data)

    # Run in parallel
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
        row = {'param': r["params"], "score": r["score"]}
        rows.append(row)

    # Minimize relative score (pred - vbs) / (sbs - vbs)
    best_score = min(rows, key=lambda x: x['score'])['score']
    equivalent_scores = [r for r in rows if math.isclose(r['score'], best_score, rel_tol=0.001)]
    best_config = min(
        equivalent_scores,
        key=lambda x: (
            x['param']['C'],
            x['param']['gamma'] if isinstance(x['param']['gamma'], (int, float)) else 0,
            0 if x['param']['kernel'] == 'rbf' else 1
        )
    )
    print('best config:', best_config)

    return best_config['param']


def test_svc(
    clf: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    par10_test: np.ndarray,
    par10_train: np.ndarray,
    solvers: list[str],
    test_data: list[dict],
    hyperparam: dict
) -> dict:
    pred = clf.predict(X_test)
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

    print(f"SVC Accuracy: {accuracy:.4f}")
    print(f"SVC Selector PAR10: {clf_par10:.2f} | SBS ({sbs_solver_name}): {sbs_par10:.2f} | VBS: {vbs_par10:.2f}")
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


def train_and_test_svc(
    train_data: list[dict],
    test_data: list[dict],
    is_wl: bool = True,
    is_wlc: bool = True,
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
    times_train = np.array([[e[s] for s in solvers] for e in train_data])
    par10_train = np.where(times_train < TIMEOUT, times_train, PAR10_PENALTY)

    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])
    times_test = np.array([[e[s] for s in solvers] for e in test_data])
    par10_test = np.where(times_test < TIMEOUT, times_test, PAR10_PENALTY)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    N_COMPONENTS_VAL = 100 if use_pca else 100000

    if not hyperparam:
        n_jobs_val = 10
        hyperparam = find_hyperparameters(X_train, y_train, par10_train, n_jobs_val)

    if len(train_data[0]['features']) > N_COMPONENTS_VAL:
        clf = Pipeline([('pca', PCA(N_COMPONENTS_VAL, random_state=42)), ('svc', SVC(**hyperparam))])
    else:
        clf = Pipeline([('minmax', MinMaxScaler()), ('svc', SVC(**hyperparam))])

    clf.fit(X_train, y_train)
    return test_svc(clf, X_test, y_test, par10_test, par10_train, solvers, test_data, hyperparam)
