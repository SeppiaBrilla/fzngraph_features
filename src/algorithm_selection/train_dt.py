import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Integer
from sklearn.tree import DecisionTreeClassifier
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn.model_selection import cross_val_score

def train_dt(X_train:np.ndarray, y_train:np.ndarray) -> dict:

    #parameters: https://www.pythonholics.com/2025/02/hyperparameter-tuning-for-decision-trees.html
    criterion = Categorical('criterion', ['gini', 'entropy', 'log_loss'], default='gini')
    max_depth = Integer('max_depth', (2, 101))
    min_sample_split = Integer('min_samples_split', (2, 10), default=5)
    min_sample_leaf = Integer('min_samples_leaf', (1, 10), default=2)

    cs = ConfigurationSpace(seed=42)
    hyprparam = [criterion, max_depth, min_sample_split, min_sample_leaf]
    cs.add(hyprparam)

    scenario = Scenario(
        cs,
        n_workers=-1,
        walltime_limit=5*60,
        n_trials=10000000
    )

    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)


    def train(config:Configuration, seed:int=42) -> float:
        config_dict = dict(config)
        if config_dict['max_depth'] == 9:
            config_dict['max_depth'] = None

        clf = DecisionTreeClassifier(**config_dict, random_state=seed)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
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

def get_dt_hyperparameteres(train_data:list[dict]) -> dict:
    X_train = np.array([e['features'] for e in train_data])
    y_train = np.array([e['label'] for e in train_data])

    hyperparam = train_dt(X_train, y_train)

    return hyperparam
