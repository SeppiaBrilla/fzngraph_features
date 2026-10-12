#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

FZN2FEAT_BIN = "/home/alessio/Documents/projects/mzn2feat/bin/fzn2feat"
DATASET_CSV = "data/sat/sat_dataset.csv"
OUTPUT_CSV = "data/sat/sat_fzn2feat.csv"

COLUMNS = [
    'c_avg_deg_cons','c_avg_dom_cons','c_avg_domdeg_cons','c_bounds_d','c_bounds_r',
    'c_bounds_z','c_cv_deg_cons','c_cv_dom_cons','c_cv_domdeg_cons','c_domain',
    'c_ent_deg_cons','c_ent_dom_cons','c_ent_domdeg_cons','c_logprod_deg_cons',
    'c_logprod_dom_cons','c_max_deg_cons','c_max_dom_cons','c_max_domdeg_cons',
    'c_min_deg_cons','c_min_dom_cons','c_min_domdeg_cons','c_num_cons','c_priority',
    'c_ratio_cons','c_sum_ari_cons','c_sum_dom_cons','c_sum_domdeg_cons','d_array_cons',
    'd_bool_cons','d_bool_vars','d_float_cons','d_float_vars','d_int_cons','d_int_vars',
    'd_ratio_array_cons','d_ratio_bool_cons','d_ratio_bool_vars','d_ratio_float_cons',
    'd_ratio_float_vars','d_ratio_int_cons','d_ratio_int_vars','d_ratio_set_cons',
    'd_ratio_set_vars','d_set_cons','d_set_vars','gc_diff_globs','gc_global_cons',
    'gc_ratio_diff','gc_ratio_globs','o_deg','o_deg_avg','o_deg_cons','o_deg_std',
    'o_dom','o_dom_avg','o_dom_deg','o_dom_std','s_bool_search','s_first_fail',
    's_goal','s_indomain_max','s_indomain_min','s_input_order','s_int_search',
    's_labeled_vars','s_other_val','s_other_var','s_set_search','v_avg_deg_vars',
    'v_avg_dom_vars','v_avg_domdeg_vars','v_cv_deg_vars','v_cv_dom_vars','v_cv_domdeg_vars',
    'v_def_vars','v_ent_deg_vars','v_ent_dom_vars','v_ent_domdeg_vars','v_intro_vars',
    'v_logprod_deg_vars','v_logprod_dom_vars','v_max_deg_vars','v_max_dom_vars',
    'v_max_domdeg_vars','v_min_deg_vars','v_min_dom_vars','v_min_domdeg_vars',
    'v_num_aliases','v_num_consts','v_num_vars','v_ratio_bounded','v_ratio_vars',
    'v_sum_deg_vars','v_sum_dom_vars','v_sum_domdeg_vars'
]

def extract_one(item):
    fzn_path, instanceid = item
    if not os.path.exists(fzn_path):
        return None
    try:
        res = subprocess.run([FZN2FEAT_BIN, fzn_path], capture_output=True, text=True, timeout=120)
        line = res.stdout.strip()
        parts = line.split(',')
        if len(parts) != len(COLUMNS):
            return None
        d = {'flatzinc': fzn_path, 'instanceid': instanceid}
        for col, val in zip(COLUMNS, parts):
            try:
                d[col] = float(val)
            except ValueError:
                d[col] = 0.0
        return d
    except Exception:
        return None

def main():
    df = pd.read_csv(DATASET_CSV)
    items = [(r['flatzinc'], r['instanceid']) for _, r in df.iterrows()]
    print(f"Extracting fzn2feat for {len(items)} SAT instances using 8 parallel workers...")

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(extract_one, item): item for item in items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="fzn2feat extraction"):
            res = future.result()
            if res:
                results.append(res)

    df_out = pd.DataFrame(results)

    # Impute any missing or timed-out instances with suite median
    missing = set(df['flatzinc']) - set(df_out['flatzinc'])
    if missing:
        print(f"Imputing {len(missing)} missing instances with column median values...")
        median_series = df_out[COLUMNS].median()
        imputed_rows = []
        for m_fzn in missing:
            m_id = df[df['flatzinc'] == m_fzn]['instanceid'].iloc[0]
            entry = {'flatzinc': m_fzn, 'instanceid': m_id}
            for col in COLUMNS:
                entry[col] = float(median_series[col])
            imputed_rows.append(entry)
        df_out = pd.concat([df_out, pd.DataFrame(imputed_rows)], ignore_index=True)

    df_out = df_out.sort_values('instanceid').reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved all {len(df_out)} instances to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
