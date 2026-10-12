#!/usr/bin/env python3
"""
extract_sat_flatzinc.py

Downloads instances from sat.inst, converts them from DIMACS CNF to MiniZinc,
and compiles them into FlatZinc (.fzn) files using MiniZinc.
"""

import os
import re
import sys
import time
import lzma
import gzip
import shutil
import urllib.request
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
URLS_FILE = ROOT_DIR / "sat.inst"
DATA_DIR = ROOT_DIR / "data" / "sat"
CNF_DIR = DATA_DIR / "cnf"
MZN_DIR = DATA_DIR / "mzn"
FZN_DIR = DATA_DIR / "fzn"
MANIFEST_FILE = DATA_DIR / "instances.csv"


def get_filename_from_url(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            cd = resp.headers.get("Content-Disposition", "")
            matches = re.findall(r'filename=([^;]+)', cd)
            if matches:
                return matches[0].strip("\"'")
    except Exception:
        pass
    # Fallback to URL hash
    return url.strip().split("/")[-1] + ".cnf.xz"


def cnf_stream_to_mzn(cnf_stream, out_stream):
    num_vars = 0
    num_clauses = 0
    current_clause = []

    for line in cnf_stream:
        line = line.strip()
        if not line or line.startswith("c"):
            continue

        if line.startswith("p cnf") or line.startswith("p"):
            parts = line.split()
            if len(parts) >= 3:
                num_vars = int(parts[2])
            if len(parts) >= 4:
                num_clauses = int(parts[3])
            out_stream.write(f"% Converted from DIMACS CNF ({num_vars} vars, {num_clauses} clauses)\n")
            out_stream.write(f"array[1..{num_vars}] of var bool: x;\n\n")
            continue

        tokens = line.split()
        for tok in tokens:
            lit = int(tok)
            if lit == 0:
                if current_clause:
                    mzn_lits = [
                        f"x[{l}]" if l > 0 else f"not x[{-l}]"
                        for l in current_clause
                    ]
                    out_stream.write(f"constraint {' \\/ '.join(mzn_lits)};\n")
                    current_clause = []
            else:
                current_clause.append(lit)

    if current_clause:
        mzn_lits = [
            f"x[{l}]" if l > 0 else f"not x[{-l}]"
            for l in current_clause
        ]
        out_stream.write(f"constraint {' \\/ '.join(mzn_lits)};\n")

    out_stream.write("\nsolve satisfy;\n")


def process_instance(url):
    """
    Downloads, converts, and compiles a single SAT instance.
    Returns (status, base_name, info_dict)
    """
    url_id = url.strip().split("/")[-1]
    
    # Check if fzn already exists
    # Look for matching fzn file starting with url_id
    existing_fzn = list(FZN_DIR.glob(f"{url_id}*.fzn"))
    if existing_fzn and existing_fzn[0].stat().st_size > 0:
        return "skipped", existing_fzn[0].stem, None

    # Step 1: Resolve filename and download
    fname = get_filename_from_url(url)
    if not fname.startswith(url_id):
        fname = f"{url_id}-{fname}"
    
    cnf_path = CNF_DIR / fname
    
    if not cnf_path.exists() or cnf_path.stat().st_size == 0:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(cnf_path, "wb") as f_out:
                shutil.copyfileobj(resp, f_out)
        except Exception as e:
            return "error_download", url_id, str(e)

    # Base name for mzn and fzn
    base_name = fname
    for ext in [".xz", ".gz", ".cnf"]:
        if base_name.endswith(ext):
            base_name = base_name[:-len(ext)]
    if not base_name.startswith(url_id):
        base_name = f"{url_id}-{base_name}"

    mzn_path = MZN_DIR / f"{base_name}.mzn"
    fzn_path = FZN_DIR / f"{base_name}.fzn"

    # Step 2: Convert to MiniZinc
    try:
        if cnf_path.suffix == ".xz":
            open_fn = lambda p: lzma.open(p, "rt", encoding="utf-8", errors="replace")
        elif cnf_path.suffix == ".gz":
            open_fn = lambda p: gzip.open(p, "rt", encoding="utf-8", errors="replace")
        else:
            open_fn = lambda p: open(p, "rt", encoding="utf-8", errors="replace")

        with open_fn(cnf_path) as f_in, open(mzn_path, "w", encoding="utf-8") as f_out:
            cnf_stream_to_mzn(f_in, f_out)
    except Exception as e:
        return "error_mzn", base_name, str(e)

    # Step 3: Compile to FlatZinc
    try:
        cmd = [
            "minizinc", "-c",
            "--solver", "chuffed",
            "--output-fzn-to-file", str(fzn_path),
            str(mzn_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            return "error_fzn", base_name, res.stderr
    except Exception as e:
        return "error_fzn", base_name, str(e)

    return "ok", base_name, None


def main():
    CNF_DIR.mkdir(parents=True, exist_ok=True)
    MZN_DIR.mkdir(parents=True, exist_ok=True)
    FZN_DIR.mkdir(parents=True, exist_ok=True)

    with open(URLS_FILE) as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Total instances in {URLS_FILE.name}: {len(urls)}")

    num_workers = min(12, os.cpu_count() or 4)
    print(f"Extracting FlatZinc using {num_workers} parallel workers...")

    results = {"ok": 0, "skipped": 0, "error": 0}
    errors = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_instance, url): url for url in urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting FlatZinc"):
            url = futures[future]
            try:
                status, base_name, err = future.result()
                if status == "ok":
                    results["ok"] += 1
                elif status == "skipped":
                    results["skipped"] += 1
                else:
                    results["error"] += 1
                    errors.append((url, status, err))
            except Exception as e:
                results["error"] += 1
                errors.append((url, "exception", str(e)))

    print("\nExtraction Summary:")
    print(f"  Compiled FlatZinc (new): {results['ok']}")
    print(f"  Already existed:         {results['skipped']}")
    print(f"  Errors:                  {results['error']}")

    if errors:
        print("\nErrors encountered:")
        for url, st, err in errors[:10]:
            print(f"  [{st}] {url}: {err}")


if __name__ == "__main__":
    main()
