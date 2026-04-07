import numpy as np
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
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
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        y_pred = np.array([self._predict(x) for x in X])
        return y_pred

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_[i] for i in k_indices]
        counts = Counter(k_nearest_labels)
        max_count = max(counts.values())
        candidates = [label for label, count in counts.items() if count == max_count]
        return max(candidates) if len(candidates) > 1 else candidates[0]

    def score(self, X, y, sample_weight=None):
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

def find_scores(X: np.ndarray, y: np.ndarray, times: np.ndarray) -> np.ndarray:
    n_features = X.shape[1]
    k = max(1, int(math.sqrt(n_features)))
    clf = KNNClassifier(k=k)
    scores = []

    for i in tqdm(range(n_features), desc="Scoring individual features"):
        X_col = X[:, i].reshape(-1, 1)
        score = cross_val_score(clf, X_col, y, times, cv=3)
        scores.append(score)

    return np.array(scores)

def _evaluate_k(k: int, X: np.ndarray, y: np.ndarray, times: np.ndarray, sorted_indices: np.ndarray) -> dict:
    np.random.seed(42)
    random.seed(42)
    clf = KNNClassifier(k=k)

    best_score = float('inf')
    best_n = 0

    for n in range(1, len(sorted_indices) + 1):
        current_indices = sorted_indices[:n]
        X_subset = X[:, current_indices]
        score = cross_val_score(clf, X_subset, y, times, cv=3)

        if score < best_score:
            best_score = score
            best_n = n
        else:
            break

    return {"k": k, "n_features": best_n, "score": best_score}

def find_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    scores: np.ndarray,
    n_jobs: int,
    ) -> dict:

    sorted_indices = np.argsort(scores)
    possible_ks = list(range(1, X.shape[0]))

    worker_fn = partial(_evaluate_k, X=X, y=y, times=times, sorted_indices=sorted_indices)

    results = []
    with Pool(processes=n_jobs) as pool:
        with tqdm(
            total=len(possible_ks),
            desc="hyperparameter search",
            unit="k",
            dynamic_ncols=True,
        ) as pbar:
            for result in pool.imap_unordered(worker_fn, possible_ks):
                results.append(result)
                pbar.set_postfix(score=f"{result['score']:.4f}", refresh=False)
                pbar.update()

    best_result = min(results, key=lambda x: x['score'])
    print('best config:', best_result)

    return {
        'k': best_result['k'],
        'n_features': best_result['n_features'],
        'selected_indices': sorted_indices[:best_result['n_features']]
    }

def test_forward_knn(clf:KNNClassifier, X_test:np.ndarray, y_test:np.ndarray, test_data:list[dict], hyperparam:dict) -> dict:
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

def train_and_test_forward_knn(train_data:list[dict], test_data:list[dict], is_wl:bool=True, is_wlc:bool=True) -> dict:
    raise NotImplementedError('forward knn has not been updated to work on borda count score')
    mp.set_start_method('spawn', force=True)

    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    times = np.array([[e['chuffed'], e['cp-sat'], e['cp-sat']] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scores = find_scores(X_train, y_train, times)

    best_config = find_hyperparameters(X_train, y_train, times, scores, n_jobs=10)

    selected_indices = best_config['selected_indices']
    best_k = best_config['k']

    X_train = X_train[:, selected_indices]
    X_test = X_test[:, selected_indices]

    clf = KNNClassifier(k=best_k)
    hyperparam = {'k': best_k, 'n_features': len(selected_indices)}
    print('hyperparameters:', hyperparam)

    clf.fit(X_train, y_train)
    return test_forward_knn(clf, X_test, y_test, test_data, hyperparam)
