import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Integer, Uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.base import clone
# from sklearn.model_selection import cross_val_score

class Preprocesser:
    def __init__(self, K):
        self.K = K
        # self.pca = PCA(K)
        self.hasher = None
        self.svd = TruncatedSVD(n_components=K, n_iter=7, random_state=42)
        self.scaler = MinMaxScaler()

    def fit(self, X):
        """Fit the transformer by identifying the indices of the top K elements."""
        _X = self.scaler.fit_transform(X)
        self.svd.fit(_X)
        # self.top_k_indices = np.argsort(-X, axis=1)[:, :self.K]
        # self.pca.fit(_X)
        return self

    def __transform(self, X):
        # if self.hasher is None:
        #     raise ValueError("The transformer has not been fitted yet. Call 'fit' first.")
        #
        # result = np.zeros((X.shape[0], self.K + 1))
        #
        # for i in range(X.shape[0]):
        #     row = X[i]
        #     top_k = row[self.hasher[i]]
        #     rest = np.delete(row, self.hasher[i])
        #     rest_sum = np.sum(rest)
        #     result[i, :self.K] = top_k
        #     result[i, self.K] = rest_sum
        # result = self.pca.transform(X)
        result = self.svd.transform(X)

        return result


    def transform(self, X):
        """Transform the input array using the fitted indices."""
        result = self.scaler.transform(X)
        result = self.__transform(result)

        return result

def cross_val_score(clf:RandomForestClassifier, X:np.ndarray, y:np.ndarray, times:np.ndarray, k:int|None, weights:np.ndarray, cv:int=5):
    scores = []
    l = len(y)
    n = l // cv
    for c in range(cv):
        _clf = clone(clf)
        X_train = np.concat((X[:n*c, :], X[n*(c+1):, :]))
        y_train = np.concat((y[:n*c], y[n*(c+1):]))
        X_test = X[n*c:n*(c+1), :]
        _times = times[n*c:n*(c+1), :]

        train_weights = np.concat((weights[:n*c], weights[n*(c+1):]))

        if k is not None:
            red = Preprocesser(k)
            red.fit(X_train)
            X_train = red.transform(X_train)
            X_test = red.transform(X_test)

        _clf.fit(X_train, y_train, sample_weight=train_weights)
        pred = _clf.predict(X_test)
        pred_time = sum([_times[i,p] for i,p in enumerate(pred)])
        t0 = sum([_times[i,0] for i,_ in enumerate(pred)])
        t1 = sum([_times[i,1] for i,_ in enumerate(pred)])
        sb_time = min(t1, t0)

        scores.append(pred_time/sb_time)
    return scores

def train_rnd_forest(X_train:np.ndarray, y_train:np.ndarray, times:np.ndarray, weights:np.ndarray) -> dict:

    #parameters: https://www.researchgate.net/figure/Tested-parameter-grid-for-random-forest-classifier_tbl1_350998771
    n_estimators = Integer('n_estimators', (200, 1001), default=200)
    criterion = Categorical('criterion', ['gini', 'entropy', 'log_loss'], default='gini')
    max_features = Categorical('max_features', ['sqrt', 'log2', None])
    max_depth = Integer('max_depth', (9, 101))
    min_sample_split = Integer('min_samples_split', (2, 10), default=5)
    min_sample_leaf = Integer('min_samples_leaf', (1, 10), default=2)
    # k_features = Integer('k', (100, 256), distribution=Uniform())

    cs = ConfigurationSpace(seed=42)
    hyprparam = [n_estimators, criterion, max_features, max_depth, min_sample_split, min_sample_leaf]#, k_features]
    cs.add(hyprparam)

    scenario = Scenario(
        cs,
        n_workers=-1,
        walltime_limit=20*60,
        n_trials=10000000
    )

    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)


    def train(config:Configuration, seed:int=42) -> float:
        config_dict = dict(config)
        if config_dict['max_depth'] == 9:
            config_dict['max_depth'] = None

        # k = config_dict['k']
        # del config_dict['k']
        clf = RandomForestClassifier(**config_dict, random_state=seed)
        scores = cross_val_score(clf, X_train, y_train, times, None, weights, cv=5)
        return float(np.mean(scores))

    smac = HyperparameterOptimizationFacade(
        scenario,
        train,
        initial_design=initial_design,
        overwrite=True,
    )
    incumbent = smac.optimize()
    if isinstance(incumbent, list):
        incumbent = incumbent[0]
    config_dict = dict(incumbent)
    if config_dict['max_depth'] == 9:
        config_dict['max_depth'] = None
    return config_dict

def test_rnd_forest(clf:RandomForestClassifier, X_test:np.ndarray, y_test:np.ndarray, test_data:list[dict], hyperparam:dict) -> dict:
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

def train_and_test_rnd_forest(train_data:list[dict], test_data:list[dict]) -> dict:
    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])
    times = np.array([[e['chuffed'], e['cp-sat'], e['cp-sat']] for e in train_data])
    X_test = np.array([e['features'] for e in test_data])
    y_test = np.array([e['label'] for e in test_data])

    weights = np.array([np.max(times[i])/np.min(times[i]) for i in range(len(times))])
    weights = 1 + (weights / np.max(weights))
    weights = np.array([1 for _ in range(len(times))])
    # preprocessor = Preprocesser(100)
    # preprocessor.fit(X_train)
    # X_train = preprocessor.transform(X_train)
    # X_test = preprocessor.transform(X_test)
    hyperparam = train_rnd_forest(X_train, y_train, times, weights)

    # k = hyperparam['k']
    # del hyperparam['k']

    clf = RandomForestClassifier(**hyperparam, random_state=42)
    print('hyperparameters:', hyperparam, 'with k:', None)
    print(np.mean(cross_val_score(clf, X_train, y_train, times, None, weights, cv=5)))

    # red = Preprocesser(k)
    # red.fit(X_train)
    # X_train = red.transform(X_train)
    # X_test = red.transform(X_test)

    clf.fit(X_train, y_train)
    # hyperparam['k'] = k
    return test_rnd_forest(clf, X_test, y_test, test_data, hyperparam)
