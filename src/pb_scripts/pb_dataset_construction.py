import glob
import os
import pandas as pd
import numpy as np

RESULTS_CSV = "data/pb25_competition_results.csv"
PORTFOLIO = [
    'OR-Tools CP-SAT 9.14 (complete)',
    'SCIP 2025-06-13 (complete)',
    'Exact 2025-06-02 (complete)',
    'SCIP-NaPS 1.00a2 (complete)',
    'UWrMaxSat-SCIP 2025-06-04 (complete)',
    'Sat4j CP 2025-06-06 (complete)',
    'Picat 2025-06-16 (complete)',
    'CASHWMaxSATDisjCom-S 2025-06-01 (complete)'
]
df_results = pd.read_csv(RESULTS_CSV)
sub = df_results[df_results['solver_name'].isin(PORTFOLIO)].copy()
solved_answers = {'OPT', 'SAT', 'UNSAT', 'OPT cert.', 'UNSAT cert.'}
sub['is_solved'] = sub['answer'].isin(solved_answers)
TIMEOUT = 1800.0
sub['effective_time'] = np.where(sub['is_solved'], sub['cpu_time'], TIMEOUT)
piv_time = sub.pivot_table(index='benchmark', columns='solver_name', values='effective_time')
files = glob.glob('data/selected-PB25/**/*.opb.xz', recursive=True) + glob.glob('data/selected-PB25/**/*.opb', recursive=True)
file_map = { (os.path.relpath(f, 'data/selected-PB25')[:-3] if f.endswith('.xz') else os.path.relpath(f, 'data/selected-PB25')): f for f in files }

instances = []
for b in piv_time.index:
    if b not in file_map:
        continue
    f = file_map[b]
    if f.endswith(".opb.xz"):
        flat = f.replace(".opb.xz", ".fzn")
    else:
        flat = f.replace(".opb", ".fzn")
    if not os.path.exists(flat):
        continue
    times = piv_time.loc[b]
    diff = times.max() - times.min()
    cat = 'OPT-LIN' if 'OPT-LIN' in b else ('DEC-LIN' if 'DEC-LIN' in b else 'OTHER')
    inst = {
        'flatzinc': flat,
        'best_solver': times.idxmin(),
        'min_time': times.min(),
        'max_time': times.max(),
    }
    for solver in PORTFOLIO:
        inst[solver] = times[solver]
    instances.append(inst)

df = pd.DataFrame(instances)
print(df.head().to_string())
df.to_csv("data/pb_dataset.csv", index=None)
