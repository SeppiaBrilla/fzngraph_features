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
    scores: np.ndarray,
    cv: int = 5,
    n_components: int = N_COMPONENTS,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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

    folds_data = []
    for train_idx, val_idx in split_gen:
        X_train_fold, X_val_fold = X[train_idx].copy(), X[val_idx].copy()
        y_train_fold = y[train_idx]
        scores_val_fold = scores[val_idx]

        n_c = min(X_train_fold.shape[0], X_train_fold.shape[1], n_components)
        if n_c > 0 and X_train_fold.shape[1] > n_c:
            pca = PCA(n_components=n_c, random_state=42)
            X_train_fold = pca.fit_transform(X_train_fold)
            X_val_fold = pca.transform(X_val_fold)

        scaler = MinMaxScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)

        folds_data.append((X_train_fold, y_train_fold, X_val_fold, scores_val_fold))

    return folds_data


def evaluate_folds_svc(params: dict, folds_data: list) -> float:
    pred_scores = []
    for X_train_fold, y_train_fold, X_val_fold, scores_val in folds_data:
        clf = SVC(**params)
        clf.fit(X_train_fold, y_train_fold)
        pred = clf.predict(X_val_fold)

        pred_score = sum([scores_val[i, p] for i, p in enumerate(pred)])
        sb_score = max([sum(scores_val[:, s_idx]) for s_idx in range(scores_val.shape[1])])
        pred_scores.append(pred_score / sb_score if sb_score > 0 else 0.0)

    return float(np.mean(pred_scores))


def cross_val_score(clf: Pipeline | SVC, X: np.ndarray, y: np.ndarray, scores: np.ndarray, cv: int = 5) -> float:
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

    pred_scores = []
    for train_idx, val_idx in split_gen:
        X_train_fold, X_val_fold = X[train_idx].copy(), X[val_idx].copy()
        y_train_fold = y[train_idx]
        scores_val_fold = scores[val_idx]

        n_c = min(X_train_fold.shape[0], X_train_fold.shape[1], N_COMPONENTS)
        steps = []
        if X_train_fold.shape[1] > n_c:
            steps.append(('pca', PCA(n_components=n_c, random_state=42)))
        svc_step = clf.named_steps['svc'] if isinstance(clf, Pipeline) else clf
        steps.append(('svc', svc_step))
        fold_clf = Pipeline(steps)

        fold_clf.fit(X_train_fold, y_train_fold)
        pred = fold_clf.predict(X_val_fold)

        pred_score = sum([scores_val_fold[i, p] for i, p in enumerate(pred)])
        sb_score = max([sum(scores_val_fold[:, s_idx]) for s_idx in range(scores_val_fold.shape[1])])
        pred_scores.append(pred_score / sb_score if sb_score > 0 else 0.0)

    return float(np.mean(pred_scores))


def _evaluate_combination(params: dict, folds_data: list) -> dict:
    np.random.seed(42)
    random.seed(42)
    score = evaluate_folds_svc(params, folds_data)
    return {"params": params, "score": score}


def find_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    n_jobs: int = mp.cpu_count(),
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

    folds_data = prepare_cv_folds(X, y, scores, cv=5)
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

    best_score = max(rows, key=lambda x: x['score'])['score']
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

    print(f"SVC Accuracy: {accuracy:.4f}")
    print(f"Selector Score: {clf_score:.2f} | SBS ({sbs_solver_name}): {sbs_score:.2f} | VBS: {vbs_score:.2f}")
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
    scores_train = np.array([[e[s] for s in solvers] for e in train_data])

    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])
    scores_test = np.array([[e[s] for s in solvers] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_c = min(X_train.shape[0], X_train.shape[1], N_COMPONENTS) if use_pca else X_train.shape[1]

    if hyperparam is None:
        hyperparam = find_hyperparameters(X_train, y_train, scores_train, mp.cpu_count())

    steps = []
    if X_train.shape[1] > n_c:
        steps.append(('pca', PCA(n_components=n_c, random_state=42)))
    steps.append(('svc', SVC(**hyperparam)))
    clf = Pipeline(steps)

    print('hyperparameters:', hyperparam)
    print("Inner CV score:", np.mean(cross_val_score(clf, X_train, y_train, scores_train, cv=3)))

    clf.fit(X_train, y_train)
    return test_svc(clf, X_test, y_test, scores_test, scores_train, solvers, test_data, hyperparam)
