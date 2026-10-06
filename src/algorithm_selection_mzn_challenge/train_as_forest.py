import numpy as np
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
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
    apply_minmax: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pre-computes and caches PCA and scaling results for each CV fold.
    This avoids re-fitting PCA repeatedly for every hyperparameter combination in parallel.
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    quantiles = np.linspace(0, 100, 8)
    gap = np.abs(scores.max(axis=1) - scores.min(axis=1))
    bins = np.unique(np.percentile(gap, quantiles))
    buckets = np.digitize(gap, bins[1:-1])

    folds_data = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, buckets)):
        X_train_fold, X_val_fold = X[train_idx].copy(), X[val_idx].copy()
        y_train_fold = y[train_idx]
        scores_val_fold = scores[val_idx]

        if X.shape[1] > n_components:
            pca = PCA(n_components, random_state=42)
            X_train_fold = pca.fit_transform(X_train_fold)
            X_val_fold = pca.transform(X_val_fold)

        if apply_minmax:
            scaler = MinMaxScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_val_fold = scaler.transform(X_val_fold)

        folds_data.append((X_train_fold, y_train_fold, X_val_fold, scores_val_fold))

    return folds_data


def evaluate_folds_forest(params: dict, folds_data: list) -> float:
    pred_scores = []
    for X_train_fold, y_train_fold, X_val_fold, scores_val in folds_data:
        clf = RandomForestClassifier(**params, class_weight={0: 1, 1: 1, 2: 1}, n_jobs=1)
        clf.fit(X_train_fold, y_train_fold)
        pred = clf.predict(X_val_fold)

        pred_score = sum([scores_val[i, p if p != 2 else 0] for i, p in enumerate(pred)])
        t0 = sum([scores_val[i, 0] for i, _ in enumerate(pred)])
        t1 = sum([scores_val[i, 1] for i, _ in enumerate(pred)])
        sb_score = max(t1, t0)

        pred_scores.append(pred_score / sb_score if sb_score != 0 else 0)

    return float(np.mean(pred_scores))


def cross_val_score(clf: Pipeline | RandomForestClassifier, X: np.ndarray, y: np.ndarray, scores: np.ndarray, cv: int = 5) -> float:
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    pred_scores = []
    quantiles = np.linspace(0, 100, 8)
    gap = np.abs(scores.max(axis=1) - scores.min(axis=1))
    bins = np.unique(np.percentile(gap, quantiles))
    buckets = np.digitize(gap, bins[1:-1])
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, buckets)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        scores_val = scores[val_idx]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        pred_score = sum([scores_val[i, p if p != 2 else 0] for i, p in enumerate(pred)])
        t0 = sum([scores_val[i, 0] for i, _ in enumerate(pred)])
        t1 = sum([scores_val[i, 1] for i, _ in enumerate(pred)])
        sb_score = max(t1, t0)

        pred_scores.append(pred_score / sb_score if sb_score != 0 else 0)

    return float(np.mean(pred_scores))


def _evaluate_combination(params: dict, folds_data: list) -> dict:
    np.random.seed(42)
    random.seed(42)
    score = evaluate_folds_forest(params, folds_data)
    return {"params": params, "score": score}


def find_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    n_jobs: int,
    is_wlc: bool = True,
) -> dict:

    param_grid = {
        'n_estimators': [n for n in range(200, 1000, 200)],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [n for n in range(10, 101, 10)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [42]
    }
    all_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(all_combinations)

    # Pre-cache PCA and scaling for each CV fold ONCE to optimize parallel performance
    folds_data = prepare_cv_folds(X, y, scores, cv=5, apply_minmax=is_wlc)

    worker_fn = partial(_evaluate_combination, folds_data=folds_data)

    # Run in parallel
    results = []
    with Pool(processes=n_jobs) as pool:
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
    best_config = min(
        equivalent_scores,
        key=lambda x: (
            x['param']['n_estimators'],
            x['param']['max_depth'] if x['param']['max_depth'] else 1000
        )
    )
    print('best config:', best_config)

    return best_config['param']


def size_evaluate(param: dict, hyperparams: dict, X: np.ndarray, y: np.ndarray, scores: np.ndarray) -> tuple[int | None, float]:
    np.random.seed(42)
    random.seed(42)
    size = param['feature_size']
    clf = RandomForestClassifier(**hyperparams, class_weight={0: 1, 1: 1, 2: 1}, n_jobs=1)
    if size is not None:
        pca = PCA(size, random_state=42)
        X_small = pca.fit_transform(X)
    else:
        X_small = X

    score = cross_val_score(clf, X_small, y, scores, 3)

    return size, score


def find_size(X: np.ndarray, y: np.ndarray, scores: np.ndarray, hyperparams: dict, is_wl: bool) -> int | None:
    param_grid = {
        'feature_size': [n for n in range(20, min(X.shape) + 1, 20)] + [None],
    }

    all_combinations = list(ParameterGrid(param_grid))

    results = []
    for comb in tqdm(all_combinations):
        results.append(size_evaluate(comb, hyperparams, X, y, scores))

    rows = []
    for r in sorted(results, key=lambda x: x[0] if x[0] else 1000):
        row = r
        rows.append(row)

    best_config = max(rows, key=lambda x: x[1])
    print('best config:', best_config)

    return best_config[0]


def test_rnd_forest(clf: Pipeline, X_test: np.ndarray, y_test: np.ndarray, test_data: list[dict], hyperparam: dict) -> dict:
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    pred_score = 0
    chuffed_score = 0
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
            pred_score += e['cp-sat']
        else:
            raise Exception(pred)
        chuffed_score += e['chuffed']
        cp_sat_score += e['cp-sat']
        vbs_score += max(e['chuffed'], e['cp-sat'])

    print(f"accuracy: {accuracy:.3f}")
    print('scores:', pred_score, chuffed_score, cp_sat_score, vbs_score)
    print(f"predicted score as a percentage of the virtual best: {vbs_score/pred_score:.3f}")
    print(f"cuffed score as a percentage of the virtual best: {vbs_score/chuffed_score:.3f}")
    print(f"cp-sat score as a percentage of the virtual best: {vbs_score/cp_sat_score:.3f}")
    print(f"predicted score as a percentage of the chuffed score: {pred_score/chuffed_score:.3f}")
    print(f"predicted score as a percentage of the cp-sat score: {pred_score/cp_sat_score:.3f}")

    return {
        'accuracy': float(accuracy),
        'clf_score': float(pred_score),
        'vbs_score': float(vbs_score),
        'chuffed_score': float(chuffed_score),
        'cp-sat_score': float(cp_sat_score),
        'clf_vbs': float(vbs_score/pred_score),
        'chuffed_vbs': float(vbs_score/chuffed_score),
        'cp-sat_vbs': float(vbs_score/cp_sat_score),
        'clf_chuffed': float(pred_score/chuffed_score),
        'clf_cp-sat': float(pred_score/cp_sat_score),
        'predictions': predictions,
        'hyperparameters': hyperparam
    }


def train_and_test_rnd_forest(train_data: list[dict], test_data: list[dict], is_wl: bool = True, is_wlc: bool = True, hyperparam: None | dict = None, use_pca: bool = True) -> dict:
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    scores = np.array([[e['cp-sat'], e['chuffed']] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    if is_wlc:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    N_COMPONENTS = 100 if use_pca else 100000
    if hyperparam is None:
        hyperparam = find_hyperparameters(X_train, y_train, scores, 10, is_wlc=is_wlc)

    if len(train_data[0]['features']) > N_COMPONENTS:
        clf = Pipeline([('pca', PCA(N_COMPONENTS, random_state=42)), ('minmax', MinMaxScaler()), ('rf', RandomForestClassifier(**hyperparam, class_weight={0: 1, 1: 1, 2: 1}))])
    else:
        clf = Pipeline([('minmax', MinMaxScaler()), ('rf', RandomForestClassifier(**hyperparam, class_weight={0: 1, 1: 1, 2: 1}))])

    print('hyperparameters:', hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, scores, cv=3)))

    clf.fit(X_train, y_train)
    return test_rnd_forest(clf, X_test, y_test, test_data, hyperparam)

