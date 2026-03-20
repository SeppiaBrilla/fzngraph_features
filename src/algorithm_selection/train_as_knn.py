import numpy as np
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.decomposition import PCA
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random, math
import multiprocessing as mp
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors Classifier compatible with scikit-learn.

    Parameters:
    -----------
    k : int, default=5
        Number of neighbors to use by default for kneighbors queries.
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns:
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        Returns:
        -------
        y : array, shape (n_samples,)
            Class labels for each data sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.array([self._predict(x) for x in X])
        return y_pred

    def _predict(self, x):
        """
        Predict the class label for a single sample x.

        Parameters:
        -----------
        x : array-like, shape (n_features,)
            A single test sample.

        Returns:
        -------
        y : int
            Class label for the sample x.
        """
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_[i] for i in k_indices]
        counts = Counter(k_nearest_labels)
        max_count = max(counts.values())
        # Get all labels with max count
        candidates = [label for label, count in counts.items() if count == max_count]
        # If there's a tie, return 1
        return max(candidates) if len(candidates) > 1 else candidates[0]

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True labels for X.
        sample_weight : array-like, shape (n_samples,), default=None
            Sample weights.

        Returns:
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

def cross_val_score(clf:KNNClassifier, X:np.ndarray, y:np.ndarray, times:np.ndarray, cv:int=5) -> float:
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
    model = KNNClassifier(**params)
    score = cross_val_score(model, X, y, times, cv=3)
    return {"params": params, "score": score}

def find_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    times:np.ndarray,
    n_jobs: int,
    ) -> dict:

    param_grid = {
        'k': list(range(1, X.shape[0])),
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
    best_config = max(equivalent_scores, key=lambda x: x['param']['k'])
    print('best config:', best_config)
 
    return best_config['param']

def size_evaluate(param:dict, hyperparams:dict, X:np.ndarray, y:np.ndarray, times:np.ndarray) -> tuple[int|None,float]:
    np.random.seed(42)
    random.seed(42)
    size = param['feature_size']
    clf = KNNClassifier(**hyperparams)
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

def test_knn(clf:KNNClassifier, X_test:np.ndarray, y_test:np.ndarray, test_data:list[dict], hyperparam:dict) -> dict:
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

def train_and_test_knn(train_data:list[dict], test_data:list[dict], is_wl:bool=True, is_wlc:bool=True) -> dict:
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
    # size = 80
    if not size is None:
        pca = PCA(n_components=size, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        min_max = MinMaxScaler()
        X_train = min_max.fit_transform(X_train)
        X_test = min_max.transform(X_test)


    clf = KNNClassifier(**hyperparam)
    print('hyperparameters:', hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, times, cv=3)))
    hyperparam['size'] = size

    clf.fit(X_train, y_train)
    return test_knn(clf, X_test, y_test, test_data, hyperparam)
