import numpy as np
from .graph_loader import load_graph
from tqdm import tqdm
from .wl_algorithms import undirected_wl_extended_features, undirected_wl_extended_features_with_edges, undirected_wl_with_node_and_edge_features, wl_extended_features, wl_extended_features_with_edges, wl_features, undirected_wl_with_node_features
from typing import Literal
from collections import Counter
import pandas as pd
import os

def prune(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict],list[dict]]:
    train_features = np.array([t['features'] for t in train_data])
    magnitude = np.sum(train_features, axis=0)
    idxs, = np.where(magnitude <= 0)
    for t in train_data:
        t['features'] = np.array(np.delete(t['features'], idxs).tolist() + [np.sum(np.array(t['features'])[idxs])])
    for t in test_data:
        t['features'] = np.array(np.delete(t['features'], idxs).tolist() + [np.sum(np.array(t['features'])[idxs])])
    return train_data, test_data

def compute_wl_features(train_data:list[dict], test_data:list[dict], wl_type:Literal['standard','node_features','edge_features','node_edge_features'], max_iter:int, all_levels:bool, undirected=False) -> tuple[list[dict],list[dict]]:
    colors = {}
    MAX_COLORS = None

    for t in tqdm(train_data, desc='train data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        if undirected:
            if wl_type == 'node_features':
                res = undirected_wl_with_node_features(g, colors, max_iter, True, MAX_COLORS)
            elif wl_type == 'node_edge_features':
                res = undirected_wl_with_node_and_edge_features(g, colors, max_iter, True, MAX_COLORS)
            else:
                raise Exception(f'unsupported undirected type {wl_type}')
        else:
            res = wl_features(g, colors, wl_type=wl_type, max_iter=max_iter, training=True, max_colors=MAX_COLORS)
        t['features'] = res

    colors_names = set(sorted(set(int(c) for c in colors.values())))
    for t in train_data:
        res = t['features']
        if all_levels:
            features = []
            for r in res:
                counter = Counter(r)
                n = len(r)
                features.extend([counter.get(color, 0) / n for color in colors_names])
        else:
            r = res[-1]
            counter = Counter(r)
            n = len(r)
            features = [counter.get(color, 0) / n for color in colors_names]
        t['features'] = features

    for t in tqdm(test_data, desc='test data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        if undirected:
            if wl_type == 'node_features':
                res = undirected_wl_with_node_features(g, colors, max_iter, False, MAX_COLORS)
            elif wl_type == 'node_edge_features':
                res = undirected_wl_with_node_and_edge_features(g, colors, max_iter, False, MAX_COLORS)
            else:
                raise Exception(f'unsupported undirected type {wl_type}')
        else:
            res = wl_features(g, colors, wl_type=wl_type, max_iter=max_iter, training=False, max_colors=MAX_COLORS)
        if all_levels:
            features = []
            for r in res:
                assert isinstance(r, list), type(r)
                counter = Counter(r)
                n = len(r)
                features.extend([counter.get(color, 0) / n for color in colors_names])
        else:
            r = res[-1]
            assert isinstance(r, list), type(r)
            counter = Counter(r)
            n = len(r)
            features = [counter.get(color, 0) / n for color in colors_names]
        t['features'] = features

    return prune(train_data, test_data)

def compute_custom_wl(train_data:list[dict], test_data:list[dict], max_iter:int, edge:bool, undirected:bool, all_levels:bool) -> tuple[list[dict],list[dict]]:
    colors = {}

    g_pairs = set()
    for t in tqdm(train_data, desc='train data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        if undirected and not edge:
            res, extra = undirected_wl_extended_features(g, colors, max_iter=max_iter, training=True)
            res = [res]
        elif undirected and edge:
            res, extra = undirected_wl_extended_features_with_edges(g, colors, max_iter=max_iter, training=True)
            res = [res]
        elif not undirected and edge:
            res, extra = wl_extended_features_with_edges(g, colors, max_iter=max_iter, training=True)
        else:
            res, extra = wl_extended_features(g, colors, max_iter=max_iter, training=True)
        t['features'] = res
        t['extra'] = extra
        for pair in extra['globals_pairs'].keys():
            g_pairs.add(pair)

    g_pairs = sorted(g_pairs)

    colors_names = set(sorted(set(int(c) for c in colors.values())))
    for t in train_data:
        res = t['features']
        if all_levels:
            features = []
            for r in res:
                counter = Counter(r)
                features.extend([counter.get(color, 0) / t['extra']['n_nodes'] for color in colors_names])
            features = np.array(features)
        else:
            r = res[-1]
            counter = Counter(r)
            features = np.array([counter.get(color, 0) / t['extra']['n_nodes'] for color in colors_names])
        tot_pairs = max(sum(t['extra']['globals_pairs'].values()), 1)
        extra = t['extra']
        t['features'] = features.tolist() + [extra['globals_pairs'].get(p,0)/tot_pairs for p in g_pairs] +\
            [extra['cpv'], extra['cpp']] #, extra['int_vars'], extra['bool_vars'], extra['set_vars'], extra['int_pars'], extra['bool_pars'], extra['set_pars'],
            #  extra['avg_dom_vars'], extra['log_search_space'], extra['log_obj_dom']]

    for t in tqdm(test_data, desc='test data'):
        with open(t['graph']) as f:
            g = load_graph(f)
        if undirected and not edge:
            res, extra = undirected_wl_extended_features(g, colors, max_iter=max_iter, training=True)
            res = [res]
        elif undirected and edge:
            res, extra = undirected_wl_extended_features_with_edges(g, colors, max_iter=max_iter, training=True)
            res = [res]
        elif not undirected and edge:
            res, extra = wl_extended_features_with_edges(g, colors, max_iter=max_iter, training=True)
        else:
            res, extra = wl_extended_features(g, colors, max_iter=max_iter, training=True)

        if all_levels:
            features = []
            for r in res:
                counter = Counter(r)
                features.extend([counter.get(color, 0) / extra['n_nodes'] for color in colors_names])
            features = np.array(features)
        else:
            r = res[-1]
            counter = Counter(r)
            features = np.array([counter.get(color, 0) / extra['n_nodes'] for color in colors_names])

        tot_pairs = max(sum(extra['globals_pairs'].values()), 1)
        t['features'] = features.tolist() + [extra['globals_pairs'].get(p,0)/tot_pairs for p in g_pairs] +\
            [extra['cpv'], extra['cpp']] #, extra['int_vars'], extra['bool_vars'], extra['set_vars'], extra['int_pars'], extra['bool_pars'], extra['set_pars'],
            #  extra['avg_dom_vars'], extra['log_search_space'], extra['log_obj_dom']]
        # 'int_vars': int_vars / n_var,
        # 'bool_vars': bool_vars / n_var,
        # 'set_vars': set_vars / n_var,
        # 'int_pars': int_pars / n_par,
        # 'bool_pars': bool_pars / n_par,
        # 'set_pars': set_pars / n_par,
    return prune(train_data, test_data)

def get_fzn2feat(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict],list[dict]]:
    fzn2feat_features = pd.read_csv('./data/fzn2feat_joined.csv')
    for t in train_data:
        d = fzn2feat_features[(fzn2feat_features['problem'] == t['model']) & (fzn2feat_features['name'] == t['name'])]
        d = d.drop(columns=['problem','name'])
        assert len(d.values) == 1, f'not one element: {d.values}, {t}'
        t['features'] = d.values[0]

    for t in test_data:
        d = fzn2feat_features[(fzn2feat_features['problem'] == t['model']) & (fzn2feat_features['name'] == t['name'])]
        d = d.drop(columns=['problem','name'])
        assert len(d.values) == 1, f'not one element: {d.values}, {t}'
        t['features'] = d.values[0]

    return train_data, test_data

def load_features(train_data:list[dict], test_data:list[dict], features_file:str) -> tuple[list[dict],list[dict]]:
    features = pd.read_csv(features_file)
    for t in train_data:
        d = features[(features['problem'] == t['model']) & (features['name'] == t['name'])]
        d = d.drop(columns=['problem','name'])
        assert len(d.values) == 1, f'not one element: {d.values}, {t}'
        t['features'] = d.values[0]

    for t in test_data:
        d = features[(features['problem'] == t['model']) & (features['name'] == t['name'])]
        d = d.drop(columns=['problem','name'])
        assert len(d.values) == 1, f'not one element: {d.values}, {t}'
        t['features'] = d.values[0]

    return train_data, test_data


def agument_features(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict], list[dict]]:
    fzn2feat_features = pd.read_csv('./data/fzn2feat_joined.csv')
    fzn2feat_cols = ['c_max_deg_cons', 'c_min_deg_cons', 'c_avg_domdeg_cons', 'v_avg_dom_vars', 'v_max_dom_vars', 'v_min_dom_vars', 'c_ent_deg_cons', 'o_dom_avg' ]
    for t in train_data:
        d = fzn2feat_features[(fzn2feat_features['problem'] == t['model']) & (fzn2feat_features['name'] == t['name'])]
        d = d[fzn2feat_cols]
        assert len(d.values) == 1, f'not one element: {d.values}, {t}'
        t['features'] = list(t['features']) + list(d.values[0])

    for t in test_data:
        d = fzn2feat_features[(fzn2feat_features['problem'] == t['model']) & (fzn2feat_features['name'] == t['name'])]
        d = d[fzn2feat_cols]
        assert len(d.values) == 1, f'not one element: {d.values}, {t}'
        t['features'] = list(t['features']) + list(d.values[0])

    return train_data, test_data

def combine(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict], list[dict]]:
    fzn2feat_features = pd.read_csv('./data/fzn2feat_joined.csv')
    for t in train_data:
        d = fzn2feat_features[(fzn2feat_features['problem'] == t['model']) & (fzn2feat_features['name'] == t['name'])]
        d = d.drop(columns=['problem','name'])
        assert len(d.values) == 1, f'not one element: {d.values}, {t}'
        t['features'] = list(t['features']) + list(d.values[0])

    for t in test_data:
        d = fzn2feat_features[(fzn2feat_features['problem'] == t['model']) & (fzn2feat_features['name'] == t['name'])]
        d = d.drop(columns=['problem','name'])
        assert len(d.values) == 1, f'not one element: {d.values}, {t}'
        t['features'] = list(t['features']) + list(d.values[0])

    return train_data, test_data

def save(train_data:list[dict], test_data:list[dict], save_name:str):
    saves = []
    for t in train_data:
        inst:dict = {i:v for i, v in enumerate(list(t['features']))}
        inst['problem'] = t['model']
        inst['name'] = t['name']
        saves.append(inst)
    for t in test_data:
        inst:dict = {i:v for i, v in enumerate(list(t['features']))}
        inst['problem'] = t['model']
        inst['name'] = t['name']
        saves.append(inst)

    pd.DataFrame(saves).to_csv(save_name, index=None)

def get_features(
        train_data:list[dict],
        test_data:list[dict],
        features_type:Literal['wlce-1', 'wlce-2', 'wlc-1', 'wlc-2', 'wlcu-1', 'wlcu-2', 'wlceu-1', 'wlceu-2', 'wl-1', 'wl-2', 'wln-1', 'wlun-1', 'wlun-2', 'wlune-1', 'wlune-2', 'wln-2', 'wle-0', 'wle-1', 'wle-2', 'wlne-1', 'wlne-2', 'fzn2feat', 'combined'],
        all_levels:bool=False,
        save_name:str="Unk"
    ) -> tuple[list[dict], list[dict]]:
    '''
    for each feature-type returns the modified train and dataset agumented with the corresponding features
    '''

    if features_type == 'wl-1':
        return compute_wl_features(train_data, test_data, 'standard', 1, all_levels)
    elif features_type == 'wl-2':
        return compute_wl_features(train_data, test_data, 'standard', 2, all_levels)

    elif features_type == 'wln-1':
        return compute_wl_features(train_data, test_data, 'node_features', 1, all_levels)
    elif features_type == 'wln-2':
        return compute_wl_features(train_data, test_data, 'node_features', 2, all_levels)

    elif features_type == 'wle-1':
        return compute_wl_features(train_data, test_data, 'edge_features', 1, all_levels)
    elif features_type == 'wle-2':
        return compute_wl_features(train_data, test_data, 'edge_features', 2, all_levels)

    elif features_type == 'wlne-1':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 1, all_levels)
    elif features_type == 'wlne-2':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 2, all_levels)

    elif features_type == 'wlun-1':
        return compute_wl_features(train_data, test_data, 'node_features', 1, False, undirected=True)
    elif features_type == 'wlun-2':
        return compute_wl_features(train_data, test_data, 'node_features', 2, False, undirected=True)

    elif features_type == 'wlune-1':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 1, False, undirected=True)
    elif features_type == 'wlune-2':
        return compute_wl_features(train_data, test_data, 'node_edge_features', 2, False, undirected=True)

    elif features_type == 'wlc-1':
        if os.path.exists(save_name):
            train, test = load_features(train_data, test_data, save_name)
        else:
            train, test = compute_custom_wl(train_data, test_data, 1, False, False, all_levels)
            save(train, test, save_name)
        # return train, test
        return agument_features(train, test)
    elif features_type == 'wlc-2':
        if os.path.exists(save_name):
            train, test = load_features(train_data, test_data, save_name)
        else:
            train, test = compute_custom_wl(train_data, test_data, 2, False, False, all_levels)
            save(train, test, save_name)
        # return train, test
        return agument_features(train, test)

    elif features_type == 'wlce-1':
        if os.path.exists(save_name):
            train, test = load_features(train_data, test_data, save_name)
        else:
            train, test = compute_custom_wl(train_data, test_data, 1, True, False, all_levels)
            save(train, test, save_name)
        # return train, test
        return agument_features(train, test)
    elif features_type == 'wlce-2':
        if os.path.exists(save_name):
            train, test = load_features(train_data, test_data, save_name)
        else:
            train, test = compute_custom_wl(train_data, test_data, 2, True, False, all_levels)
            save(train, test, save_name)
        # return train, test
        return agument_features(train, test)

    elif features_type == 'wlcu-1':
        return compute_custom_wl(train_data, test_data, 1, False, True, all_levels)
    elif features_type == 'wlcu-2':
        return compute_custom_wl(train_data, test_data, 2, False, True, all_levels)

    elif features_type == 'wlceu-1':
        return compute_custom_wl(train_data, test_data, 1, False, True, all_levels)
    elif features_type == 'wlceu-2':
        return compute_custom_wl(train_data, test_data, 2, False, True, all_levels)

    elif features_type =='combined':
        train, test = compute_custom_wl(train_data, test_data, 1, False, False, all_levels)
        return combine(train, test)

    elif features_type == 'fzn2feat':
        return get_fzn2feat(train_data, test_data)

    raise Exception(f'unsupported features type {features_type}')
