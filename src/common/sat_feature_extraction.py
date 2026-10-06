import os
import re
import json
import uuid
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim


class Loss:
    def __init__(self, alpha:Tensor, beta:Tensor):
        self.alpha = alpha
        self.beta = beta

    def global_loss(self, z:Tensor, z_hat:Tensor) -> Tensor:
        res = torch.tensor(0)
        for i in range(z.shape[0]):
            res += torch.norm(z[i] - z_hat[i], 2) ** 2
        return res

    def local_loss(self, y_i:Tensor, y_j:Tensor) -> Tensor:
        res = torch.tensor(0)
        for i in range(y_i.shape[0]):
            for j in range(y_j.shape[0]):
                res += torch.norm(y_i[i] - y_j[j], 2) ** 2
        return res

def parse_cnf(cnf_path):
    if not os.path.exists(cnf_path):
        return 0, 0, []
    clauses = []
    num_vars = 0
    with open(cnf_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('p cnf'):
                parts = line.split()
                if len(parts) >= 3:
                    num_vars = int(parts[2])
            elif line and not line.startswith(('c', 'p', '%', '=')):
                parts = line.split()
                clause = [int(p) for p in parts if int(p) != 0]
                if clause:
                    clauses.append(clause)
    num_clauses = len(clauses)
    return num_vars, num_clauses, clauses

def compute_S_plus_fast(num_vars, num_clauses, clauses, lam=0.15, mu=0.85):
    row_ind_pos, col_ind_pos = [], []
    row_ind_neg, col_ind_neg = [], []

    for i, c in enumerate(clauses):
        for lit in c:
            var = abs(lit) - 1
            if var < num_vars:
                if lit > 0:
                    row_ind_pos.append(i)
                    col_ind_pos.append(var)
                else:
                    row_ind_neg.append(i)
                    col_ind_neg.append(var)
                    
    L_pos = sp.csr_matrix((np.ones(len(row_ind_pos), dtype=np.float32), (row_ind_pos, col_ind_pos)), shape=(num_clauses, num_vars))
    L_neg = sp.csr_matrix((np.ones(len(row_ind_neg), dtype=np.float32), (row_ind_neg, col_ind_neg)), shape=(num_clauses, num_vars))
    L = sp.hstack([L_pos, L_neg], format='csr')
    
    S = (L_pos + L_neg).tocsr()
    S.data = np.ones_like(S.data, dtype=np.float32)
    
    # 1. d_c = diag(L @ L.T) is row sum of L
    d_c = np.array(L.sum(axis=1)).flatten()
    d_c[d_c == 0] = 1.0
    inv_sqrt_d_c = 1.0 / np.sqrt(d_c)
    
    # v_c = P_dir_c @ 1 = D_c^{-1/2} @ L @ (L.T @ (D_c^{-1/2} @ 1))
    w_c = inv_sqrt_d_c
    u_c = L.T @ w_c
    v_c = inv_sqrt_d_c * (L @ u_c)
    
    # 2. d_x = diag(S.T @ S) is column sum of S
    d_x = np.array(S.sum(axis=0)).flatten()
    d_x[d_x == 0] = 1.0
    inv_sqrt_d_x = 1.0 / np.sqrt(d_x)
    
    # v_x = P_dir_x @ 1 = D_x^{-1/2} @ S.T @ (S @ (D_x^{-1/2} @ 1))
    w_x = inv_sqrt_d_x
    u_x = S @ w_x
    v_x = inv_sqrt_d_x * (S.T @ u_x)
    
    # Term 1: lambda * S * (v_c / 2m + v_x / 2n)
    S_coo = S.tocoo()
    rows, cols = S_coo.row, S_coo.col
    term1_data = lam * S_coo.data * (v_c[rows] / (2.0 * max(num_clauses, 1)) + v_x[cols] / (2.0 * max(num_vars, 1)))
    Term1 = sp.csr_matrix((term1_data, (rows, cols)), shape=(num_clauses, num_vars))
    
    # Term 2: (mu / 2) * (P_indir_c @ S + S @ P_indir_x)
    # Using matrix associativity to avoid any M x M allocations:
    d_S_c = np.array(S.sum(axis=1)).flatten()
    d_S_c[d_S_c == 0] = 1.0
    inv_sqrt_d_Sc = 1.0 / np.sqrt(d_S_c)
    
    D_Sc_inv = sp.diags(inv_sqrt_d_Sc)
    S_tilde = D_Sc_inv @ S
    
    M_A = S_tilde.T @ S   # (N x N)
    T2_A = S_tilde @ M_A  # (M x N)
    
    D_x_inv = sp.diags(inv_sqrt_d_x)
    S_hat = S @ D_x_inv
    
    M_B = S_hat.T @ S_hat # (N x N)
    T2_B = S @ M_B        # (M x N)
    
    Term2 = (mu / 2.0) * (T2_A + T2_B)
    S_plus = (Term1 + Term2).tocsr()
    return S_plus

def extract_features_for_cnf(cnf_path):
    num_vars, num_clauses, clauses = parse_cnf(cnf_path)
    if num_clauses == 0 or num_vars == 0:
        return np.zeros(150)

    try:
        S_plus = compute_S_plus_fast(num_vars, num_clauses, clauses)
    except Exception as e:
        print(f"Error computing S_plus: {e}")
        return np.zeros(150)

    S_T = S_plus.T.tocsr()
    S = S_plus.tocsr()

    max_dim = max(num_clauses, num_vars)
    num_nodes = num_clauses + num_vars

    try:
        class DPGR(nn.Module):
            def __init__(self, input_dim, hidden_dim=150):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, hidden_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, input_dim)
                )
            def forward(self, x):
                h = self.encoder(x)
                out = self.decoder(h)
                return out, h

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DPGR(max_dim, 150).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Adaptive batch size to prevent OOM on large instances
        batch_size = max(16, min(256, int(30000000 / max(max_dim, 1))))
        
        def get_batch(indices):
            batch_data = np.zeros((len(indices), max_dim), dtype=np.float32)
            clause_mask = indices < num_clauses
            var_mask = ~clause_mask
            
            if np.any(clause_mask):
                c_idx = indices[clause_mask]
                S_plus_c = S[c_idx, :]
                batch_data[clause_mask, :num_vars] = S_plus_c.toarray() if sp.issparse(S_plus_c) else S_plus_c
                
            if np.any(var_mask):
                v_idx = indices[var_mask] - num_clauses
                S_plus_v = S_T[v_idx, :]
                batch_data[var_mask, :num_clauses] = S_plus_v.toarray() if sp.issparse(S_plus_v) else S_plus_v
                
            return torch.tensor(batch_data, device=device)

        model.train()
        epochs = 5
        max_steps_per_epoch = 100
        for epoch in range(epochs):
            indices = np.random.permutation(num_nodes)
            steps = min(max_steps_per_epoch, (num_nodes + batch_size - 1) // batch_size)
            for s in range(steps):
                batch_idx = indices[s*batch_size : (s+1)*batch_size]
                Z_batch = get_batch(batch_idx)
                
                optimizer.zero_grad()
                out, _ = model(Z_batch)
                loss = criterion(out, Z_batch)
                loss.backward()
                optimizer.step()
                
        model.eval()
        embeddings = []
        with torch.no_grad():
            eval_steps = min(100, (num_nodes + batch_size - 1) // batch_size)
            for s in range(eval_steps):
                batch_idx = np.arange(s*batch_size, min((s+1)*batch_size, num_nodes))
                Z_batch = get_batch(batch_idx)
                _, h = model(Z_batch)
                embeddings.append(h.cpu().numpy())
                
        if embeddings:
            h_all = np.vstack(embeddings)
            graph_emb = h_all.mean(axis=0)
            return graph_emb
        else:
            return np.zeros(150)
        
    except (ImportError, Exception) as e:
        # Fallback to TruncatedSVD
        from sklearn.decomposition import TruncatedSVD
        n_components = min(150, S.shape[1]-1, S.shape[0]-1)
        if n_components <= 0:
            return np.zeros(150)
        svd = TruncatedSVD(n_components=n_components)
        h_c = svd.fit_transform(S)
        h_v = svd.components_.T
        h = np.vstack([h_c, h_v])
        graph_emb = h.mean(axis=0)
        res = np.zeros(150)
        res[:len(graph_emb)] = graph_emb
        return res

def get_mzn_dzn_paths(year, problem, instance):
    base_dir = f"../ucloudExecutor/mzn-challenge/{year}/{problem}"
    if not os.path.isdir(base_dir):
        return None, None
    models = [f for f in os.listdir(base_dir) if f.endswith(".mzn")]
    if not models:
        return None, None
    model = os.path.join(base_dir, models[0])
    
    if instance:
        dzn = os.path.join(base_dir, f"{instance}.dzn")
        if not os.path.exists(dzn):
            dzn = os.path.join(base_dir, f"{instance}.json")
            if not os.path.exists(dzn):
                return model, None
        return model, dzn
    return model, None

def generate_cnf(model_path, data_path, cache_dir="data/.cache_cnf"):
    os.makedirs(cache_dir, exist_ok=True)
    base_name = os.path.basename(model_path) + "_" + (os.path.basename(data_path) if data_path else "none")
    cnf_path = os.path.join(cache_dir, base_name + ".cnf")
    if os.path.exists(cnf_path):
        return cnf_path
        
    cwd = f"/tmp/{uuid.uuid4()}"
    os.makedirs(cwd, exist_ok=True)
    tmp_mzn = os.path.join(cwd, "model.mzn")
    with open(tmp_mzn, 'w') as f:
        f.write(f'include "{os.path.abspath(model_path)}";\n')
        
    fzn_path = os.path.join(cwd, "instance.fzn")
    cmd_mzn = ["minizinc", "-c", "--solver", "picat", tmp_mzn]
    if data_path:
        cmd_mzn.append(os.path.abspath(data_path))
    cmd_mzn.extend(["-o", fzn_path])
    
    try:
        subprocess.run(cmd_mzn, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
    except Exception:
        pass
        
    if os.path.exists(fzn_path):
        # find picat script
        pi_file = "/home/alessio/.local/opt/fzn_picat/fzn_picat_sat.pi"
        cmd_picat = ["picat", pi_file, "dumpcnf", fzn_path]
        try:
            subprocess.run(cmd_picat, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
        except Exception:
            pass
            
    tmp_cnf = os.path.join(cwd, "__tmp.cnf")
    if os.path.exists(tmp_cnf) and os.path.getsize(tmp_cnf) > 0:
        os.rename(tmp_cnf, cnf_path)
    else:
        with open(cnf_path, 'w') as f:
            f.write("p cnf 0 0\n")
            
    # cleanup tmp dir
    try:
        for f_name in os.listdir(cwd):
            os.remove(os.path.join(cwd, f_name))
        os.rmdir(cwd)
    except Exception:
        pass
        
    return cnf_path

def load_features(train_data, test_data, features_file):
    features = pd.read_csv(features_file)
    if 'year' in features.columns:
        features['year'] = features['year'].astype(int)
    for t in train_data:
        inst_col = 'instance' if 'instance' in features.columns else 'name'
        if 'year' in features.columns:
            d = features[(features['problem'] == t['model']) & (features['year'] == int(t['year'])) & (features[inst_col] == t['name'])]
            d = d.drop(columns=['problem','year',inst_col])
        else:
            d = features[(features['problem'] == t['model']) & (features[inst_col] == t['name'])]
            d = d.drop(columns=['problem',inst_col])
        t['features'] = d.values[0]

    for t in test_data:
        inst_col = 'instance' if 'instance' in features.columns else 'name'
        if 'year' in features.columns:
            d = features[(features['problem'] == t['model']) & (features['year'] == int(t['year'])) & (features[inst_col] == t['name'])]
            d = d.drop(columns=['problem','year',inst_col])
        else:
            d = features[(features['problem'] == t['model']) & (features[inst_col] == t['name'])]
            d = d.drop(columns=['problem',inst_col])
        t['features'] = d.values[0]
    return train_data, test_data

def save_features(train_data, test_data, save_name):
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    saves = []
    for t in train_data + test_data:
        inst = {i: v for i, v in enumerate(list(t['features']))}
        inst['problem'] = t['model']
        inst['year'] = t['year']
        inst['instance'] = t['name']
        saves.append(inst)
    pd.DataFrame(saves).to_csv(save_name, index=None)

def ensure_picat_patched():
    try:
        res = subprocess.run(["minizinc", "--solvers"], capture_output=True, text=True)
        paths = []
        in_search = False
        for line in res.stdout.split('\n'):
            if line.startswith("Search path for solver"):
                in_search = True
                continue
            if in_search and line.strip().startswith("/"):
                paths.append(line.strip())
                
        msc_path = None
        for p in paths:
            candidate = os.path.join(p, "picat.msc")
            if os.path.exists(candidate):
                msc_path = candidate
                break
                
        if not msc_path:
            return
            
        with open(msc_path, 'r') as f:
            msc = json.load(f)
            
        executable = msc.get("executable", "")
        if not os.path.exists(executable):
            return
            
        with open(executable, 'r') as f:
            sh_content = f.read()
            
        pi_file = None
        for token in sh_content.split():
            if token.endswith(".pi"):
                pi_file = token
                break
                
        if not pi_file or not os.path.exists(pi_file):
            return
            
        with open(pi_file, 'r') as f:
            pi_content = f.read()
            
        if "dumpcnf" not in pi_content:
            pi_content = pi_content.replace('process_args(["-a"|As],File) =>', 'process_args(["dumpcnf"|As],File) =>\n    get_heap_map().put(dump_cnf,1),\n    process_args(As,File).\nprocess_args(["-a"|As],File) =>')
            
            p1_old = 'proc_solve(all,_LabelCalls,PVars,_SVars,ROutAnns,Options) ?=>\n    solve(Options,PVars),'
            p1_new = 'proc_solve(all,_LabelCalls,PVars,_SVars,ROutAnns,Options) ?=>\n    (get_heap_map().has_key(dump_cnf) -> solve($[dump("__tmp.cnf")|Options],PVars), halt ; true),\n    solve(Options,PVars),'
            pi_content = pi_content.replace(p1_old, p1_new)
            
            p2_old = 'proc_solve(one,_LabelCalls,PVars,_SVars,ROutAnns,Options) =>\n    (solve(Options,PVars) ->'
            p2_new = 'proc_solve(one,_LabelCalls,PVars,_SVars,ROutAnns,Options) =>\n    (get_heap_map().has_key(dump_cnf) -> solve($[dump("__tmp.cnf")|Options],PVars), halt ; true),\n    (solve(Options,PVars) ->'
            pi_content = pi_content.replace(p2_old, p2_new)
            
            p3_old = 'proc_solve(min(Obj),_LabelCalls,PVars,_SVars,ROutAnns,Options) =>\n    (solve($[min(Obj),report(fzn_output_obj(ROutAnns,Obj))|Options],PVars) ->'
            p3_new = 'proc_solve(min(Obj),_LabelCalls,PVars,_SVars,ROutAnns,Options) =>\n    (get_heap_map().has_key(dump_cnf) -> solve($[dump("__tmp.cnf"),min(Obj),report(fzn_output_obj(ROutAnns,Obj))|Options],PVars), halt ; true),\n    (solve($[min(Obj),report(fzn_output_obj(ROutAnns,Obj))|Options],PVars) ->'
            pi_content = pi_content.replace(p3_old, p3_new)
            
            p4_old = 'proc_solve(max(Obj),_LabelCalls,PVars,_SVars,ROutAnns,Options) =>\n    (solve($[max(Obj),report(fzn_output_obj(ROutAnns,Obj))|Options],PVars) ->'
            p4_new = 'proc_solve(max(Obj),_LabelCalls,PVars,_SVars,ROutAnns,Options) =>\n    (get_heap_map().has_key(dump_cnf) -> solve($[dump("__tmp.cnf"),max(Obj),report(fzn_output_obj(ROutAnns,Obj))|Options],PVars), halt ; true),\n    (solve($[max(Obj),report(fzn_output_obj(ROutAnns,Obj))|Options],PVars) ->'
            pi_content = pi_content.replace(p4_old, p4_new)
            
            with open(pi_file, 'w') as f:
                f.write(pi_content)
    except Exception:
        pass

def get_sat_features_pipeline(train_data, test_data, save_name):
    ensure_picat_patched()
    if os.path.exists(save_name):
        return load_features(train_data, test_data, save_name)
        
    all_data = train_data + test_data
    for t in tqdm(all_data, desc="Extracting SAT features"):
        model_path, data_path = get_mzn_dzn_paths(t['year'], t['model'], t['instance'] if t.get('instance') else t['name'])
        if model_path:
            cnf_path = generate_cnf(model_path, data_path)
            feats = extract_features_for_cnf(cnf_path)
        else:
            feats = np.zeros(150)
        t['features'] = feats.tolist()
        
    save_features(train_data, test_data, save_name)
    return train_data, test_data
