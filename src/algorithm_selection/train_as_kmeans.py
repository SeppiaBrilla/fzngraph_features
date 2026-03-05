import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

class K_means_classifier(BaseEstimator):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kmeans = KMeans(**kwargs)
        self.rnd_state = kwargs['random_state']
        self.clusters_sbs = None

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        clusters = self.kmeans.fit_predict(X)
        sbss = {c: {0:0, 1:1} for c in range(self.kmeans.get_params()['n_clusters'])}
        for i in range(len(clusters)):
            cluster = clusters[i]
            label = y[i]
            sbss[cluster][label if label != 2 else 0] += 1
        self.clusters_sbs = {c: max(sbss[c].items(), key=lambda x: x[1])[0] for c in range(self.kmeans.get_params()['n_clusters'])}

    def predict(self, X:np.ndarray) -> np.ndarray:
        if self.clusters_sbs is None:
            raise Exception('kmeans classfier not trained')
        try:
            preds = self.kmeans.predict(X)
            res = []
            for p in preds:
                res.append(self.clusters_sbs[int(p)])
            return np.array(res)
        except Exception as e:
            print(self.clusters_sbs)
            raise e

    def get_params(self, deep=True):
        return self.kmeans.get_params(deep)

    def set_params(self, **params):
        self.kmeans.set_params(**params)
        return self

class TopKReducer:
    def __init__(self, K):
        self.K = K
        self.pca = PCA(K)
        self.top_k_indices = None
        self.scaler = MinMaxScaler()

    def fit(self, X):
        """Fit the transformer by identifying the indices of the top K elements."""
        self.top_k_indices = np.argsort(-X, axis=1)[:, :self.K]
        # _X = self.pca.fit_transform(X)
        self.scaler.fit(self.__transform(X))
        return self

    def __transform(self, X):
        if self.top_k_indices is None:
            raise ValueError("The transformer has not been fitted yet. Call 'fit' first.")

        result = np.zeros((X.shape[0], self.K + 1))

        for i in range(X.shape[0]):
            row = X[i]
            top_k = row[self.top_k_indices[i]]
            rest = np.delete(row, self.top_k_indices[i])
            rest_sum = np.sum(rest)
            result[i, :self.K] = top_k
            result[i, self.K] = rest_sum
        # result = self.pca.transform(X)

        return result


    def transform(self, X):
        """Transform the input array using the fitted indices."""
        result = self.__transform(X)
        result = self.scaler.transform(result)

        return result

def cross_val_score(clf:K_means_classifier, X:np.ndarray, y:np.ndarray, times:np.ndarray, k:int|None, cv:int=5):
    scores = []
    l = len(y)
    n = l // cv
    for c in range(cv):
        _clf = clone(clf)
        X_train = np.concat((X[:n*c, :],X[n*(c+1):, :]))
        y_train = np.concat((y[:n*c],y[n*(c+1):]))
        X_test = X[n*c:n*(c+1), :]
        _times = times[n*c:n*(c+1), :]

        red = TopKReducer(X_train.shape[1])
        red.fit(X_train)
        X_train = red.transform(X_train)
        X_test = red.transform(X_test)

        _clf.fit(X_train, y_train)
        pred = _clf.predict(X_test)
        pred_time = sum([_times[i,p] for i,p in enumerate(pred)])
        t0 = sum([_times[i,0] for i,_ in enumerate(pred)])
        t1 = sum([_times[i,1] for i,_ in enumerate(pred)])
        sb_time = min(t0, t1)

        scores.append(pred_time/sb_time)
    return scores

def train_kmeans(X_train:np.ndarray, y_train:np.ndarray, times:np.ndarray, reduce:bool) -> dict:
    parameters = list(ParameterGrid({
            'n_clusters': range(2, 21),
            'init': ['k-means++', 'random'],
            'max_iter': [100, 200, 300],
            'tol': [1e-3, 1e-4, 1e-5],
            'n_init': [5, 10, 15, "auto"],
            'random_state': [42],
            'verbose': [0]
        }))

    res = []
    for par in tqdm(parameters):
        scores = cross_val_score(K_means_classifier(**par), X_train, y_train, times, None)
        res.append(np.mean(scores))
    idx = np.argmin(res)
    return parameters[idx]

def test_kmean(clf:K_means_classifier, X_test:np.ndarray, y_test:np.ndarray, test_data:list[dict]) -> dict:
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

    print(f"predicted time as a percentage of the virtual best: {pred_time/vbs_time:.3f}")
    print(f"cuffed time as a percentage of the virtual best: {chuffed_time/vbs_time:.3f}")
    print(f"cp-sat time as a percentage of the virtual best: {cp_sat_time_time/vbs_time:.3f}")
    print(f"predicted time as a percentage of the chuffed time: {pred_time/chuffed_time:.3f}")
    print(f"predicted time as a percentage of the cp-sat time: {pred_time/cp_sat_time_time:.3f}")

    return {
        'accuracy': accuracy,
        'clf_vbs': pred_time/vbs_time,
        'chuffed_vbs': chuffed_time/vbs_time,
        'cp-sat_vbs': cp_sat_time_time/vbs_time,
        'clf_chuffed': pred_time/chuffed_time,
        'clf_cp-sat': pred_time/cp_sat_time_time
        }

def train_and_test_kmeans(train_data:list[dict], test_data:list[dict], reduce:bool) -> dict:
    # scaler = MinMaxScaler()
    X_train = np.array([e['features'] for e in train_data])
    # X_train = scaler.fit_transform(X_train)
    y_train = np.array([e['label'] for e in train_data])
    times = np.array([[e['chuffed'], e['cp-sat'], e['chuffed']] for e in train_data])
    # class P:
    #     def predict(self, X:np.ndarray):
    #         return [1 for _ in range(X.shape[0])]
    #     def fit(self, X, Y):
    #         pass
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])
    # print(cross_val_score(P(), X_test, y_test, times, None,  10))
    # raise Exception()
    hyperparam = train_kmeans(X_train, y_train, times, reduce)

    k = None
    if 'k' in hyperparam:
        k = hyperparam['k']
        del hyperparam['k']

    # X_test = scaler.transform(X_test)

    clf = K_means_classifier(**hyperparam)
    print(np.mean(cross_val_score(clf, X_train, y_train, times, k, cv=5)))

    if reduce:
        red = TopKReducer(k)
        red.fit(X_train)
        X_train = red.transform(X_train)
        X_test = red.transform(X_test)

    clf.fit(X_train, y_train)
    return test_kmean(clf, X_test, y_test, test_data)
