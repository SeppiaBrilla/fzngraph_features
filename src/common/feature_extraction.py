import numpy as np
from .graph_loader import load_graph
from tqdm import tqdm
from .wl_algorithms import undirected_wl_extended_features, undirected_wl_extended_features_with_edges, undirected_wl_with_node_and_edge_features, wl_extended_features, wl_extended_features_with_edges, wl_features, undirected_wl_with_node_features
from typing import Literal
from collections import Counter
import pandas as pd
import os
import gc

def prune(train_data:list[dict], test_data:list[dict]) -> tuple[list[dict],list[dict]]:
    train_features = np.array([t['features'] for t in train_data])
    magnitude = np.sum(train_features, axis=0)
    idxs, = np.where(magnitude <= 0)
    for t in train_data:
        t['features'] = np.array(np.delete(t['features'], idxs).tolist() + [np.sum(np.array(t['features'])[idxs])])
    for t in test_data:
        t['features'] = np.array(np.delete(t['features'], idxs).tolist() + [np.sum(np.array(t['features'])[idxs])])
    return train_data, test_data

def compute_wl_features(train_data:list[dict], test_data:list[dict], wl_type:Literal['standard','node_features','edge_features','node_edge_features'], max_iter:int, all_levels:bool, undirected=False, colors_file:str='colors.bin') -> tuple[list[dict],list[dict]]:
    os.makedirs(os.path.dirname(os.path.abspath(colors_file)), exist_ok=True)

    for t in tqdm(train_data, desc='train data'):
        graph_input = t.get('graph', t.get('flatzinc'))
        if undirected:
            if wl_type == 'node_features':
                res = undirected_wl_with_node_features(graph_input, colors_file, max_iter, True)
            elif wl_type == 'node_edge_features':
                res = undirected_wl_with_node_and_edge_features(graph_input, colors_file, max_iter, True)
            else:
                raise Exception(f'unsupported undirected type {wl_type}')
        else:
            res = wl_features(graph_input, colors_file, wl_type=wl_type, max_iter=max_iter, training=True)
        t['color_counts'] = res

    colors_names = sorted(list(set(color for t in train_data for color in t['color_counts'].keys())))
    for t in train_data:
        c_counts = t['color_counts']
        n = sum(c_counts.values()) or 1
        t['features'] = [c_counts.get(color, 0) / n for color in colors_names]

    for t in tqdm(test_data, desc='test data'):
        graph_input = t.get('graph', t.get('flatzinc'))
        if undirected:
            if wl_type == 'node_features':
                res = undirected_wl_with_node_features(graph_input, colors_file, max_iter, False)
            elif wl_type == 'node_edge_features':
                res = undirected_wl_with_node_and_edge_features(graph_input, colors_file, max_iter, False)
            else:
                raise Exception(f'unsupported undirected type {wl_type}')
        else:
            res = wl_features(graph_input, colors_file, wl_type=wl_type, max_iter=max_iter, training=False)
        c_counts = res
        n = sum(c_counts.values()) or 1
        t['features'] = [c_counts.get(color, 0) / n for color in colors_names]

    return prune(train_data, test_data)

def compute_custom_wl(train_data:list[dict], test_data:list[dict], max_iter:int, edge:bool, undirected:bool, all_levels:bool, colors_file:str) -> tuple[list[dict],list[dict]]:
    g_pairs = set()
    for i, t in tqdm(enumerate(train_data), desc='train data', total=len(train_data)):
        graph_input = t.get('graph', t.get('flatzinc'))
        is_train = True
        if undirected and not edge:
            color_counts, extra = undirected_wl_extended_features(graph_input, colors_file, max_iter=max_iter, training=is_train)
        elif undirected and edge:
            color_counts, extra = undirected_wl_extended_features_with_edges(graph_input, colors_file, max_iter=max_iter, training=is_train)
        elif not undirected and edge:
            color_counts, extra = wl_extended_features_with_edges(graph_input, colors_file, max_iter=max_iter, training=is_train)
        else:
            color_counts, extra = wl_extended_features(graph_input, colors_file, max_iter=max_iter, training=is_train)
        t['color_counts'] = color_counts
        t['extra'] = extra
        for pair in extra['globals_pairs'].keys():
            g_pairs.add(pair)
        gc.collect()

    g_pairs = sorted(g_pairs)

    colors_names = sorted(list(set(color for t in train_data for color in t['color_counts'].keys())))
    for t in train_data:
        n_nodes = max(t['extra']['n_nodes'], 1)
        c_counts = t['color_counts']
        color_feats = [c_counts.get(color, 0) / n_nodes for color in colors_names]
        
        tot_pairs = max(sum(t['extra']['globals_pairs'].values()), 1)
        extra = t['extra']
        pair_feats = [extra['globals_pairs'].get(p, 0) / tot_pairs for p in g_pairs]
        
        extra_scalar_feats = [
            extra['cpv'],
            extra['cpp'],
            extra['d_ratio_int_vars'],
            extra['d_ratio_bool_vars'],
            extra['o_deg_cons'],
            extra['o_deg_std'],
            extra['o_dom_deg'],
            extra['v_ent_deg_vars'],
            extra['v_sum_domdeg_vars']
        ]
        extra_scalar_feats = np.nan_to_num(extra_scalar_feats, nan=0.0, posinf=0.0, neginf=0.0).tolist()
        t['features'] = color_feats + pair_feats + extra_scalar_feats

    for t in tqdm(test_data, desc='test data'):
        graph_input = t.get('graph', t.get('flatzinc'))
        if undirected and not edge:
            color_counts, extra = undirected_wl_extended_features(graph_input, colors_file, max_iter=max_iter, training=False)
        elif undirected and edge:
            color_counts, extra = undirected_wl_extended_features_with_edges(graph_input, colors_file, max_iter=max_iter, training=False)
        elif not undirected and edge:
            color_counts, extra = wl_extended_features_with_edges(graph_input, colors_file, max_iter=max_iter, training=False)
        else:
            color_counts, extra = wl_extended_features(graph_input, colors_file, max_iter=max_iter, training=False)

        n_nodes = max(extra['n_nodes'], 1)
        color_feats = [color_counts.get(color, 0) / n_nodes for color in colors_names]
        
        tot_pairs = max(sum(extra['globals_pairs'].values()), 1)
        pair_feats = [extra['globals_pairs'].get(p, 0) / tot_pairs for p in g_pairs]
        
        extra_scalar_feats = [
            extra['cpv'],
            extra['cpp'],
            extra['d_ratio_int_vars'],
            extra['d_ratio_bool_vars'],
            extra['o_deg_cons'],
            extra['o_deg_std'],
            extra['o_dom_deg'],
            extra['v_ent_deg_vars'],
            extra['v_sum_domdeg_vars']
        ]
        extra_scalar_feats = np.nan_to_num(extra_scalar_feats, nan=0.0, posinf=0.0, neginf=0.0).tolist()
        t['features'] = color_feats + pair_feats + extra_scalar_feats

    return prune(train_data, test_data)




def save(train_data:list[dict], test_data:list[dict], save_name:str):
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    saves = []
    for dataset in (train_data, test_data):
        for t in dataset:
            inst:dict = {i:v for i, v in enumerate(list(t['features']))}
            if 'flatzinc' in t:
                inst['flatzinc'] = t['flatzinc']
            if 'model' in t:
                inst['problem'] = t['model']
            if 'year' in t:
                inst['year'] = t['year']
            if 'name' in t:
                inst['instance'] = t['name']
            saves.append(inst)

    pd.DataFrame(saves).to_csv(save_name, index=None)


