import os
from typing import Literal
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.feature_extraction import (
    compute_wl_features,
    compute_custom_wl,
    save,
)

DEFAULT_FZN2FEAT_PATH = os.path.join(os.path.dirname(__file__), '../../data/fzn2feat_mzn_challenge.csv')


def load_mzn_features(train_data: list[dict], test_data: list[dict], features_file: str) -> tuple[list[dict], list[dict]]:
    """
    Loads features for MiniZinc Challenge instances using the ('model', 'year', 'name') key.
    """
    features = pd.read_csv(features_file)
    meta_cols = {'flatzinc', 'problem', 'year', 'instance', 'name', 'model'}
    feat_cols = [c for c in features.columns if c not in meta_cols]
    inst_col = 'instance' if 'instance' in features.columns else 'name'
    prob_col = 'problem' if 'problem' in features.columns else 'model'

    lookup = {}
    for _, row in features.iterrows():
        key = (str(row[prob_col]), int(row['year']), str(row[inst_col]))
        lookup[key] = row[feat_cols].values.astype(float)

    for t in train_data:
        key = (str(t['model']), int(t['year']), str(t['name']))
        if key not in lookup:
            raise KeyError(f"Could not find features for {key} in {features_file}")
        t['features'] = lookup[key]

    for t in test_data:
        key = (str(t['model']), int(t['year']), str(t['name']))
        if key not in lookup:
            raise KeyError(f"Could not find features for {key} in {features_file}")
        t['features'] = lookup[key]

    return train_data, test_data


def get_fzn2feat(train_data: list[dict], test_data: list[dict], fzn2feat_path: str = DEFAULT_FZN2FEAT_PATH) -> tuple[list[dict], list[dict]]:
    """
    Loads pre-extracted fzn2feat features for MiniZinc Challenge instances.
    """
    if not os.path.exists(fzn2feat_path):
        raise FileNotFoundError(f"fzn2feat features file not found at {fzn2feat_path}")

    df = pd.read_csv(fzn2feat_path)
    meta_cols = {'problem', 'year', 'instance', 'name', 'model', 'flatzinc'}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    inst_col = 'instance' if 'instance' in df.columns else 'name'
    prob_col = 'problem' if 'problem' in df.columns else 'model'

    # Clean inf and nan
    df[feat_cols] = df[feat_cols].fillna(0)
    df[feat_cols] = df[feat_cols].replace([np_inf := float('inf'), -np_inf], float('nan'))
    max_vals = df[feat_cols].max().fillna(0)
    df[feat_cols] = df[feat_cols].fillna(max_vals).fillna(0)

    lookup = {}
    for _, row in df.iterrows():
        key = (str(row[prob_col]), int(row['year']), str(row[inst_col]))
        lookup[key] = row[feat_cols].values.astype(float)

    for t in train_data:
        key = (str(t['model']), int(t['year']), str(t['name']))
        t['features'] = lookup[key]

    for t in test_data:
        key = (str(t['model']), int(t['year']), str(t['name']))
        t['features'] = lookup[key]

    return train_data, test_data


def get_features(
    train_data: list[dict],
    test_data: list[dict],
    features_type: str,
    all_levels: bool = False,
    save_name: str = "Unk"
) -> tuple[list[dict], list[dict]]:
    """
    Feature extraction pipeline for the MiniZinc Challenge dataset.
    """
    if os.path.exists(save_name):
        return load_mzn_features(train_data, test_data, save_name)

    bin_file = save_name.replace('.csv', '.bin') if save_name != "Unk" else "colors.bin"

    if features_type == 'wl-1':
        train, test = compute_wl_features(train_data, test_data, 'standard', 1, all_levels, colors_file=bin_file)
    elif features_type == 'wl-2':
        train, test = compute_wl_features(train_data, test_data, 'standard', 2, all_levels, colors_file=bin_file)
    elif features_type == 'wln-1':
        train, test = compute_wl_features(train_data, test_data, 'node_features', 1, all_levels, colors_file=bin_file)
    elif features_type == 'wln-2':
        train, test = compute_wl_features(train_data, test_data, 'node_features', 2, all_levels, colors_file=bin_file)
    elif features_type == 'wle-1':
        train, test = compute_wl_features(train_data, test_data, 'edge_features', 1, all_levels, colors_file=bin_file)
    elif features_type == 'wle-2':
        train, test = compute_wl_features(train_data, test_data, 'edge_features', 2, all_levels, colors_file=bin_file)
    elif features_type == 'wlne-1':
        train, test = compute_wl_features(train_data, test_data, 'node_edge_features', 1, all_levels, colors_file=bin_file)
    elif features_type == 'wlne-2':
        train, test = compute_wl_features(train_data, test_data, 'node_edge_features', 2, all_levels, colors_file=bin_file)
    elif features_type == 'wlun-1':
        train, test = compute_wl_features(train_data, test_data, 'node_features', 1, False, undirected=True, colors_file=bin_file)
    elif features_type == 'wlun-2':
        train, test = compute_wl_features(train_data, test_data, 'node_features', 2, False, undirected=True, colors_file=bin_file)
    elif features_type == 'wlune-1':
        train, test = compute_wl_features(train_data, test_data, 'node_edge_features', 1, False, undirected=True, colors_file=bin_file)
    elif features_type == 'wlune-2':
        train, test = compute_wl_features(train_data, test_data, 'node_edge_features', 2, False, undirected=True, colors_file=bin_file)

    elif features_type == 'wlc-1':
        train, test = compute_custom_wl(train_data, test_data, 1, False, False, all_levels, colors_file=bin_file)
    elif features_type == 'wlc-2':
        train, test = compute_custom_wl(train_data, test_data, 2, False, False, all_levels, colors_file=bin_file)
    elif features_type == 'wlce-1':
        train, test = compute_custom_wl(train_data, test_data, 1, True, False, all_levels, colors_file=bin_file)
    elif features_type == 'wlce-2':
        train, test = compute_custom_wl(train_data, test_data, 2, True, False, all_levels, colors_file=bin_file)

    elif features_type == 'wlcu-1':
        train, test = compute_custom_wl(train_data, test_data, 1, False, True, all_levels, colors_file=bin_file)
    elif features_type == 'wlcu-2':
        train, test = compute_custom_wl(train_data, test_data, 2, False, True, all_levels, colors_file=bin_file)
    elif features_type == 'wlceu-1':
        train, test = compute_custom_wl(train_data, test_data, 1, True, True, all_levels, colors_file=bin_file)
    elif features_type == 'wlceu-2':
        train, test = compute_custom_wl(train_data, test_data, 2, True, True, all_levels, colors_file=bin_file)

    elif features_type == 'fzn2feat':
        return get_fzn2feat(train_data, test_data)

    else:
        raise Exception(f'unsupported features type {features_type}')

    if save_name != "Unk":
        save(train, test, save_name)
    return train, test
