import json, os
import pandas as pd
from copy import deepcopy
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score

order = {
    k:i for i,k in enumerate(['fzn2feat', 'wl-1', 'wl-2', 'wle-1', 'wle-2', 'wln-1', 'wln-2', 'wlne-1', 'wlne-2', 'wlc-1', 'wlc-2', 'wlce-1', 'wlce-2', 'majority_classifier'])
}

def get_as_results(result_folder:str) -> dict[str,dict[str, float]]:
    '''
    Function to load algorithm selection (scores) results from a specific folder.
    '''
    n_fold = 2
    as_results = {}
    for file in os.listdir(result_folder):
        if not file.startswith("as_"):
            continue
        if 'forest' in file:
            name_split = file.replace("as_", "").replace("-rnd-forest", "").replace("-forest", "").replace(".json", "").split("-")
        elif 'svc' in file:
            name_split = file.replace("as_", "").replace("-svc", "").replace("_", "-").replace(".json", "").split("-")
        elif 'nn' in file:
            name_split = file.replace("as_", "").replace("-nn", "").replace(".json", "").split("-")
        fold = "-".join(name_split[-n_fold:])
        seed = name_split[-(n_fold-1)]
        assert seed in ["7", '12', "42", "72", "123", "156", "197", "205", "224", "242"], (seed, file)
        feature_names = "-".join(name_split[:-n_fold])
        # print(name_split, fold, feature_names)
        with open(os.path.join(result_folder, file)) as f:
            content = json.load(f)
        if feature_names not in as_results:
            as_results[feature_names] = {}
        as_results[feature_names][fold] = (content['clf_score'] - content['cp-sat_score']) / (content['vbs_score'] - content['cp-sat_score'])
    return {k:v for k,v in sorted(as_results.items(), key= lambda x: order[x[0]])}

def get_parallel_results(result_folder):
    '''
    Function to load parallel results from a specific folder.
    '''
    as_results = {}
    for file in os.listdir(result_folder):
        if not file.startswith("parallelise_"):
            continue
        if 'forest' in file:
            name_split = file.replace("parallelise_", "").replace("-rnd-forest", "").replace("-forest", "").replace(".json", "").split("-")
        elif 'svc' in file:
            name_split = file.replace("parallelise_", "").replace("-svc", "").replace(".json", "").split("-")
        elif 'nn' in file:
            name_split = file.replace("parallelise_", "").replace("-nn", "").replace(".json", "").split("-")
        fold = name_split[-1]
        feature_names = "-".join(name_split[:-1])
        with open(os.path.join(result_folder, file)) as f:
            content = json.load(f)
        if feature_names not in as_results:
            as_results[feature_names] = {}
        as_results[feature_names][fold] = content['accuracy']
    return {k:v for k,v in sorted(as_results.items(), key= lambda x: (order[x[0].replace('-max', '').replace('-min', '')], 0 if '-min' in x[0] else 1))}

def to_table(data, wsbs=True):
    '''
    transforms data loaded from a folder to a table. It works also if results from different folders are merged together
    '''
    df = pd.DataFrame.from_dict(data, orient='index')

    df = df[sorted(df.columns, key=str)]

    data_cols = df.columns
    if wsbs:
        df['Worse_sbs'] = df.apply(lambda row: (row < 0).sum(), axis=1)
    df['Mean']      = df[data_cols].mean(axis=1)
    df['Median']    = df[data_cols].median(axis=1)
    df['Std']       = df[data_cols].std(axis=1)
    df['Max']       = df[data_cols].max(axis=1)
    df['Min']       = df[data_cols].min(axis=1)
    return df[['Mean', 'Median', 'Std', 'Max', 'Min'] + (['Worse_sbs'] if wsbs else [])]

def get_as_accuracy_results(result_folder):
    as_results = {}
    for file in os.listdir(result_folder):
        if not file.startswith("as-accuracy_"):
            continue
        if 'forest' in file:
            name_split = file.replace("as-accuracy_", "").replace("-rnd-forest", "").replace("-forest", "").replace(".json", "").split("-")
        elif 'svc' in file:
            name_split = file.replace("as-accuracy_", "").replace("-svc", "").replace(".json", "").split("-")
        elif 'nn' in file:
            name_split = file.replace("as-accuracy_", "").replace("-nn", "").replace(".json", "").split("-")
        fold = "-".join(name_split[-2:])
        feature_names = "-".join(name_split[:-2])
        try:
            with open(os.path.join(result_folder, file)) as f:
                content = json.load(f)
        except Exception as e:
            print(file)
            raise e
        if feature_names not in as_results:
            as_results[feature_names] = {}
        as_results[feature_names][fold] = content['accuracy']
    return {k:v for k,v in sorted(as_results.items(), key= lambda x: order[x[0]])}

def combine_algorithm_selection_score(result_1:str, result_2:str, dataset:str, result_3:str|None=None) -> float:
    dataset = pd.read_csv(dataset)
    with open(result_1) as f:
        r1 = json.load(f)
    with open(result_2) as f:
        r2 = json.load(f)
    if result_3:
        with open(result_3) as f:
            r3 = json.load(f)

    score = 0
    sb_score = 0
    vbs_score = 0
    for k in r1['predictions'].keys():
        
        [model, name] = k.split("-sep-")
        row = dataset[((dataset['name'] == name) & (dataset['model'] == model))]
        cp_sat_score = row['cp-sat'].values[0]
        chuffed_score = row['chuffed'].values[0]
        cplex_score = row['cplex'].values[0]
        inst_scores = [cp_sat_score, chuffed_score, cplex_score]
        
        if result_3:
            preds = [0, 0, 0]
            preds[r1['predictions'][k]] += 1
            preds[r2['predictions'][k]] += 1
            preds[r3['predictions'][k]] += 1
            pred = np.argmax(preds)
            score += inst_scores[pred]
        else:
            score += inst_scores[r1['predictions'][k]] if r1['predictions'][k] == r2['predictions'][k] else cp_sat_score 
        sb_score += cp_sat_score
        vbs_score += max(inst_scores)
    assert vbs_score == r1['vbs_score'] and vbs_score == r2['vbs_score'], f'different vbs scores {[vbs_score, r1['vbs_score'], r2['vbs_score']]}'
    assert sb_score == r1['cp-sat_score'] and sb_score == r2['cp-sat_score'], f'different sb scores {[sb_score, r1['cp-sat_score'], r2['cp-sat_score']]}'
    return (score - r1['cp-sat_score']) / (r1['vbs_score'] - r1['cp-sat_score'])

def compute_mcnemar(results: list[dict[str,int]]) -> float:
    """
    Compute McNemar's test p-value comparing model predictions vs baseline.

    Args:
        results: list of dicts with keys 'p1', 'p2', 'true'

    Returns:
        p-value from McNemar's test
    """
    correct_model    = [int(r['p1']     == r['true']) for r in results]
    correct_baseline = [int(r['p2'] == r['true']) for r in results]

    # Disagreement counts
    b = sum(cm == 0 and cb == 1 for cm, cb in zip(correct_model, correct_baseline))  # baseline right, model wrong
    c = sum(cm == 1 and cb == 0 for cm, cb in zip(correct_model, correct_baseline))  # model right, baseline wrong

    # 2x2 contingency table:
    #         model correct | model wrong
    # base correct  [a, b]
    # base wrong    [c, d]
    a = sum(cm == 1 and cb == 1 for cm, cb in zip(correct_model, correct_baseline))
    d = sum(cm == 0 and cb == 0 for cm, cb in zip(correct_model, correct_baseline))

    table = [[a, b], [c, d]]

    # Use exact=True when b+c < 25, Yates correction otherwise
    exact = (b + c) < 25
    result = mcnemar(table, exact=exact, correction=not exact)

    return result.pvalue

def compare_as_accuracy_results(comb1:tuple[str, str], comb2:tuple[str, str], base_folder:str) -> tuple[float, float, float]:
    '''
    Given two combinations of (features, model) compares their result using the mcnemar P value and returns the total accuracy of each combination
    '''
    features1, model1 = comb1
    folder = os.path.join(base_folder, f"{model1}-scores")
    res = {}

    # Load results for the first combination
    for file in os.listdir(folder):
        with open(os.path.join(folder, file)) as f:
            content = json.load(f)
        split = file.replace("as-accuracy_","").replace(".json", "").split('-')
        seed = split[-1]
        fold = split[-2]
        files_features = split[0] if split[0] == 'fzn2feat' else f'{split[0]}-{split[1]}'

        # If the file does not correspond to the features we are looking for, we skip it.
        if features1 != files_features:
            continue

        # Load all predictions and create the key-value pairs into the results dict if the don't exist 
        for k, v in content['predictions'].items():
            if not seed in res:
                res[seed] = {}
            if not k in res[seed]:
                res[seed][k] = {'true': v['true']}
            res[seed][k][features1] = v['pred']
    
    features2, model2 = comb2
    folder = os.path.join(base_folder, f"{model2}-scores")

    # Load results for the second combination
    for file in os.listdir(folder):
        with open(os.path.join(folder, file)) as f:
            content = json.load(f)
        split = file.replace("as-accuracy_","").replace(".json", "").split('-')
        seed = split[-1]
        fold = split[-2]
        files_features = split[0] if split[0] == 'fzn2feat' else f'{split[0]}-{split[1]}'

        # If the file does not correspond to the features we are looking for, we skip it.
        if features2 != files_features:
            continue

        # Load all predictions and check they correspond
        for k, v in content['predictions'].items():
            assert res[seed][k]['true'] == v['true']
            res[seed][k][features2] = v['pred']

    c1 = []
    c2 = []
    trues = []
    comparison_dicts = []

    # Reshape the results to be in line with the format required by the functions
    for v in res.values():
        for preds in v.values():
            trues.append(preds['true'])
            c1.append(preds[features1])
            c2.append(preds[features2])

            comparison_dicts.append({'p1': preds[features1], 'p2': preds[features2], 'true': preds['true']})
    
    return accuracy_score(trues, c1), accuracy_score(trues, c2), compute_mcnemar(comparison_dicts)

if __name__ == '__main__':
    import sys, pprint
    r = get_as_results(sys.argv[1])
    df = to_table(r, False)
    pprint.pp(df)

    print('wlc only, svc fold 0 seed 7:',
        combine_algorithm_selection_score('data/as-scores-results/svc-scores/as_wlc-1-svc-0-7.json', 'data/as-scores-results/svc-scores/as_wlc-2-svc-0-7.json')
    )
    
    print('wlc and fzn2feat, svc fold 0 seed 7:',
        combine_algorithm_selection_score('data/as-scores-results/svc-scores/as_wlc-1-svc-0-7.json', 'data/as-scores-results/svc-scores/as_wlc-2-svc-0-7.json', 'data/as-scores-results/svc-scores/as_fzn2feat-svc-0-7.json')
    )