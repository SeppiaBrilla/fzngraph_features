from train_dt import get_dt_hyperparameteres
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


def feature_cross_validate(clf:DecisionTreeClassifier, X:np.ndarray, y:np.ndarray, times:np.ndarray, k:int=5, cv:int=5) -> list[tuple[float, np.ndarray]]:
    n_el = X.shape[0]
    el_per_bucket = n_el // cv
    scores = []
    for _cv in range(cv):
        _clf = clone(clf)
        X_train = np.concat((X[:el_per_bucket*_cv, :], X[el_per_bucket*(_cv+1):, :]))
        y_train = np.concat((y[:el_per_bucket*_cv], y[el_per_bucket*(_cv+1):]))
        X_test = X[el_per_bucket*_cv:el_per_bucket*(_cv+1), :]
        _times = times[el_per_bucket*_cv:el_per_bucket*(_cv+1), :]

        assert isinstance(_clf, DecisionTreeClassifier)
        _clf.fit(X_train, y_train)
        pred = _clf.predict(X_test)
        pred_time = sum([_times[i,p] for i,p in enumerate(pred)])
        t0 = sum([_times[i,0] for i,_ in enumerate(pred)])
        t1 = sum([_times[i,1] for i,_ in enumerate(pred)])
        sb_time = min(t1, t0)

        clf_importance = _clf.feature_importances_
        idxs = [int(idx) for idx in np.argpartition(clf_importance, k)[-k:]]
        importance = [idx for idx in idxs if clf_importance[idx] > 0]
        scores.append((
            pred_time/sb_time,
            importance
        ))
    return scores


def feature_selection(train_data:list[dict], k:int=5, cv:int=5) -> np.ndarray:
    hyperparams = get_dt_hyperparameteres(train_data)
    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    times = np.array([[e['chuffed'], e['cp-sat'], e['cp-sat']] for e in train_data])
    clf = DecisionTreeClassifier(**hyperparams)
    _s = set()
    for s in feature_cross_validate(clf, X_train, y_train, times, k, cv):
        for f in s[1]:
            _s.add(f)

    return np.array(list(_s))
