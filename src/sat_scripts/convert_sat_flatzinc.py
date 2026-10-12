#!/usr/bin/env python3
"""
convert_sat_flatzinc.py

Downloads instances from sat.inst, converts DIMACS CNF to MiniZinc,
and compiles to FlatZinc (.fzn) using MiniZinc with:
- RAM limit: 10 GB (discard if exceeded)
- Timeout: 20 minutes (1200 seconds, discard if exceeded)
- Discarded/failed instances recorded to data/sat/discarded_instances.json
- Converted instances recorded to data/sat/converted_instances.json
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
import resource
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
URLS_FILE = ROOT_DIR / "sat.inst"
DATA_DIR = ROOT_DIR / "data" / "sat"
CNF_DIR = DATA_DIR / "cnf"
MZN_DIR = DATA_DIR / "mzn"
FZN_DIR = DATA_DIR / "fzn"
DISCARDED_FILE = DATA_DIR / "discarded_instances.json"
CONVERTED_FILE = DATA_DIR / "converted_instances.json"

MAX_RAM_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
TIMEOUT_SECONDS = 20 * 60                # 20 minutes = 1200 seconds


def set_process_limits():
    """Sets virtual memory (address space) limit to 10 GB."""
    resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM_BYTES, MAX_RAM_BYTES))


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
    Returns:
        dict: {
            'url': url,
            'url_id': url_id,
            'status': 'ok' | 'skipped' | 'discarded',
            'reason': reason_str if discarded else None,
            'fzn_path': str if ok/skipped else None,
            'elapsed_s': float
        }
    """
    t0 = time.time()
    url_id = url.strip().split("/")[-1]

    # Check if valid fzn already exists
    existing_fzn = list(FZN_DIR.glob(f"{url_id}*.fzn"))
    if existing_fzn and existing_fzn[0].stat().st_size > 0:
        return {
            'url': url,
            'url_id': url_id,
            'status': 'skipped',
            'reason': None,
            'fzn_path': str(existing_fzn[0]),
            'elapsed_s': time.time() - t0
        }

    # Step 1: Download CNF if not present
    existing_cnfs = list(CNF_DIR.glob(f"{url_id}*"))
    if existing_cnfs and existing_cnfs[0].stat().st_size > 0:
        cnf_path = existing_cnfs[0]
        fname = cnf_path.name
    else:
        fname = get_filename_from_url(url)
        if not fname.startswith(url_id):
            fname = f"{url_id}-{fname}"
        cnf_path = CNF_DIR / fname
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(cnf_path, "wb") as f_out:
                shutil.copyfileobj(resp, f_out)
        except Exception as e:
            return {
                'url': url,
                'url_id': url_id,
                'status': 'discarded',
                'reason': f"Download error: {str(e)}",
                'fzn_path': None,
                'elapsed_s': time.time() - t0
            }

    # Base name for mzn and fzn
    base_name = fname
    for ext in [".xz", ".gz", ".cnf"]:
        if base_name.endswith(ext):
            base_name = base_name[:-len(ext)]
    if not base_name.startswith(url_id):
        base_name = f"{url_id}-{base_name}"

    mzn_path = MZN_DIR / f"{base_name}.mzn"
    fzn_path = FZN_DIR / f"{base_name}.fzn"

    # Step 2: Convert to MiniZinc if not already created
    if not mzn_path.exists() or mzn_path.stat().st_size == 0:
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
            return {
                'url': url,
                'url_id': url_id,
                'status': 'discarded',
                'reason': f"CNF-to-MZN conversion error: {str(e)}",
                'fzn_path': None,
                'elapsed_s': time.time() - t0
            }

    # Step 3: Compile to FlatZinc with 10 GB RAM limit and 20 min timeout
    cmd = [
        "minizinc", "-c",
        "--solver", "chuffed",
        "--output-fzn-to-file", str(fzn_path),
        str(mzn_path)
    ]

    try:
        t_compile_start = time.time()
        res = subprocess.run(
            cmd,
            preexec_fn=set_process_limits,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )
        compile_elapsed = time.time() - t_compile_start

        if res.returncode != 0:
            # Check if terminated by signal (e.g. SIGKILL / SIGSEGV from out of memory)
            if fzn_path.exists():
                fzn_path.unlink()
            err_msg = res.stderr.strip() or f"Process exited with code {res.returncode}"
            if res.returncode in (-9, -11, 137):
                reason = f"OOM exceeded 10GB limit (signal {res.returncode})"
            else:
                reason = f"Compilation failed ({err_msg})"
            return {
                'url': url,
                'url_id': url_id,
                'status': 'discarded',
                'reason': reason,
                'fzn_path': None,
                'elapsed_s': time.time() - t0
            }

    except subprocess.TimeoutExpired:
        if fzn_path.exists():
            fzn_path.unlink()
        return {
            'url': url,
            'url_id': url_id,
            'status': 'discarded',
            'reason': f"Compilation timeout after {TIMEOUT_SECONDS}s (20 minutes)",
            'fzn_path': None,
            'elapsed_s': time.time() - t0
        }
    except Exception as e:
        if fzn_path.exists():
            fzn_path.unlink()
        return {
            'url': url,
            'url_id': url_id,
            'status': 'discarded',
            'reason': f"Compilation error: {str(e)}",
            'fzn_path': None,
            'elapsed_s': time.time() - t0
        }

    # Verify output file
    if not fzn_path.exists() or fzn_path.stat().st_size == 0:
        if fzn_path.exists():
            fzn_path.unlink()
        return {
            'url': url,
            'url_id': url_id,
            'status': 'discarded',
            'reason': "Empty or missing FlatZinc output",
            'fzn_path': None,
            'elapsed_s': time.time() - t0
        }

    return {
        'url': url,
        'url_id': url_id,
        'status': 'ok',
        'reason': None,
        'fzn_path': str(fzn_path),
        'elapsed_s': time.time() - t0
    }


def main():
    CNF_DIR.mkdir(parents=True, exist_ok=True)
    MZN_DIR.mkdir(parents=True, exist_ok=True)
    FZN_DIR.mkdir(parents=True, exist_ok=True)

    with open(URLS_FILE) as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Total instances in {URLS_FILE.name}: {len(urls)}")

    # Load existing records if any
    discarded = {}
    if DISCARDED_FILE.exists():
        try:
            with open(DISCARDED_FILE) as f:
                discarded = json.load(f)
        except Exception:
            discarded = {}

    converted = {}
    if CONVERTED_FILE.exists():
        try:
            with open(CONVERTED_FILE) as f:
                converted = json.load(f)
        except Exception:
            converted = {}

    # Filter out already discarded or converted
    pending_urls = []
    for u in urls:
        uid = u.strip().split("/")[-1]
        if uid in discarded:
            continue
        # Also check if already in fzn
        existing_fzn = list(FZN_DIR.glob(f"{uid}*.fzn"))
        if existing_fzn and existing_fzn[0].stat().st_size > 0:
            converted[uid] = str(existing_fzn[0])
            continue
        pending_urls.append(u)

    print(f"Already converted: {len(converted)}")
    print(f"Already discarded: {len(discarded)}")
    print(f"Pending to process: {len(pending_urls)}")

    # Use 2 parallel workers to avoid exceeding 30 GB system RAM when each worker can use up to 10 GB
    num_workers = 2
    print(f"Processing with {num_workers} parallel workers (max 10GB RAM per worker, 20min timeout)...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_instance, url): url for url in pending_urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting SAT to FlatZinc"):
            res = future.result()
            uid = res['url_id']
            if res['status'] in ('ok', 'skipped'):
                converted[uid] = res['fzn_path']
                print(f"[SUCCESS] {uid} ({res['elapsed_s']:.1f}s)")
            else:
                discarded[uid] = {
                    'url': res['url'],
                    'reason': res['reason'],
                    'elapsed_s': res['elapsed_s']
                }
                print(f"[DISCARDED] {uid}: {res['reason']} ({res['elapsed_s']:.1f}s)")

            # Save progress incrementally
            with open(CONVERTED_FILE, "w") as f:
                json.dump(converted, f, indent=2)
            with open(DISCARDED_FILE, "w") as f:
                json.dump(discarded, f, indent=2)

    print("\n================ FINAL SUMMARY ================")
    print(f"Total Successfully Converted: {len(converted)}")
    print(f"Total Discarded:             {len(discarded)}")
    print(f"Converted list saved to:     {CONVERTED_FILE}")
    print(f"Discarded list saved to:     {DISCARDED_FILE}")


if __name__ == "__main__":
    main()
