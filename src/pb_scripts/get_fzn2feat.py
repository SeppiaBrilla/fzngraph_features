import glob
import os
import pandas as pd
import subprocess
from tqdm import tqdm

COLUMNS = ['c_avg_deg_cons','c_avg_dom_cons','c_avg_domdeg_cons','c_bounds_d','c_bounds_r','c_bounds_z','c_cv_deg_cons','c_cv_dom_cons','c_cv_domdeg_cons','c_domain','c_ent_deg_cons','c_ent_dom_cons','c_ent_domdeg_cons','c_logprod_deg_cons','c_logprod_dom_cons','c_max_deg_cons','c_max_dom_cons','c_max_domdeg_cons','c_min_deg_cons','c_min_dom_cons','c_min_domdeg_cons','c_num_cons','c_priority','c_ratio_cons','c_sum_ari_cons','c_sum_dom_cons','c_sum_domdeg_cons','d_array_cons','d_bool_cons','d_bool_vars','d_float_cons','d_float_vars','d_int_cons','d_int_vars','d_ratio_array_cons','d_ratio_bool_cons','d_ratio_bool_vars','d_ratio_float_cons','d_ratio_float_vars','d_ratio_int_cons','d_ratio_int_vars','d_ratio_set_cons','d_ratio_set_vars','d_set_cons','d_set_vars','gc_diff_globs','gc_global_cons','gc_ratio_diff','gc_ratio_globs','o_deg','o_deg_avg','o_deg_cons','o_deg_std','o_dom','o_dom_avg','o_dom_deg','o_dom_std','s_bool_search','s_first_fail','s_goal','s_indomain_max','s_indomain_min','s_input_order','s_int_search','s_labeled_vars','s_other_val','s_other_var','s_set_search','v_avg_deg_vars','v_avg_dom_vars','v_avg_domdeg_vars','v_cv_deg_vars','v_cv_dom_vars','v_cv_domdeg_vars','v_def_vars','v_ent_deg_vars','v_ent_dom_vars','v_ent_domdeg_vars','v_intro_vars','v_logprod_deg_vars','v_logprod_dom_vars','v_max_deg_vars','v_max_dom_vars','v_max_domdeg_vars','v_min_deg_vars','v_min_dom_vars','v_min_domdeg_vars','v_num_aliases','v_num_consts','v_num_vars','v_ratio_bounded','v_ratio_vars','v_sum_deg_vars','v_sum_dom_vars','v_sum_domdeg_vars']
files = glob.glob('data/selected-PB25/**/*.opb.xz', recursive=True) + glob.glob('data/selected-PB25/**/*.opb', recursive=True)

instances = []
for file in tqdm(files):
    if file.endswith(".opb.xz"):
        flat = file.replace(".opb.xz", ".fzn")
    else:
        flat = file.replace(".opb", ".fzn")
    if not os.path.exists(flat):
        continue

    inst = {
        'flatzinc': flat,
    }
    res = subprocess.run([f'/home/alessio/Documents/projects/mzn2feat/bin/fzn2feat {flat}'], shell=True, stdout=subprocess.PIPE)
    feats = res.stdout.decode().strip()
    for i, feat in enumerate(feats.split(',')):
        inst[COLUMNS[i]] = feat

    instances.append(inst)

df = pd.DataFrame(instances)
df.to_csv("data/pb_fzn2feat.csv", index=None)
