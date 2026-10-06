import os
from typing import Literal
import pandas as pd
from common.feature_extraction import (
    compute_wl_features,
    compute_custom_wl,
    save,
)


def load_mzn_features(train_data: list[dict], test_data: list[dict], features_file: str) -> tuple[list[dict], list[dict]]:
    """
    Loads features for MiniZinc Challenge instances strictly using (problem, year, instance) composite key.
    """
    features = pd.read_csv(features_file)
    meta_cols = {'problem', 'year', 'instance', 'name'}
    feat_cols = [c for c in features.columns if c not in meta_cols]
    inst_col = 'instance' if 'instance' in features.columns else 'name'

    lookup = {}
    for _, row in features.iterrows():
        key = (str(row['problem']), int(row['year']), str(row[inst_col]))
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


def get_fzn2feat(train_data: list[dict], test_data: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Loads fzn2feat features using the (problem, year, instance) key for MiniZinc Challenge.
    """
    return load_mzn_features(train_data, test_data, './data/fzn2feat_mzn_challenge.csv')


def combine(train_data: list[dict], test_data: list[dict]) -> tuple[list[dict], list[dict]]:
    features = pd.read_csv('./data/fzn2feat_new.csv').fillna(0)
    meta_cols = {'problem', 'year', 'instance', 'name'}
    feat_cols = [c for c in features.columns if c not in meta_cols]
    inst_col = 'instance' if 'instance' in features.columns else 'name'

    lookup = {}
    for _, row in features.iterrows():
        key = (str(row['problem']), int(row['year']), str(row[inst_col]))
        lookup[key] = row[feat_cols].values.astype(float).tolist()

    for t in train_data:
        key = (str(t['model']), int(t['year']), str(t['name']))
        t['features'] = list(t['features']) + lookup[key]

    for t in test_data:
        key = (str(t['model']), int(t['year']), str(t['name']))
        t['features'] = list(t['features']) + lookup[key]

    return train_data, test_data


def get_features(
    train_data: list[dict],
    test_data: list[dict],
    features_type: Literal[
        'wlce-1', 'wlce-2', 'wlc-1', 'wlc-2', 'wlcu-1', 'wlcu-2', 'wlceu-1', 'wlceu-2',
        'wl-1', 'wl-2', 'wln-1', 'wlun-1', 'wlun-2', 'wlune-1', 'wlune-2', 'wln-2',
        'wle-0', 'wle-1', 'wle-2', 'wlne-1', 'wlne-2', 'fzn2feat', 'combined', 'sat-features'
    ],
    all_levels: bool = False,
    save_name: str = "Unk"
) -> tuple[list[dict], list[dict]]:
    """
    Feature extraction pipeline for the MiniZinc Challenge dataset.
    """
    # Check if precomputed features exist
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

    elif features_type == 'combined':
        train, test = compute_custom_wl(train_data, test_data, 1, False, False, all_levels, colors_file=bin_file)
        return combine(train, test)

    elif features_type == 'sat-features':
        from common.sat_feature_extraction import get_sat_features_pipeline
        return get_sat_features_pipeline(train_data, test_data, save_name)

    elif features_type == 'fzn2feat':
        return get_fzn2feat(train_data, test_data)

    else:
        raise Exception(f'unsupported features type {features_type}')

    if save_name != "Unk":
        save(train, test, save_name)
    return train, test
