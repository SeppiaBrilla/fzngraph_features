#!/usr/bin/env python3
"""
cnf2mzn.py

Converts a DIMACS CNF file (plain or .xz/.gz compressed) into a MiniZinc (.mzn) file.

Usage:
    python3 cnf2mzn.py input.cnf[.xz/.gz] [-o output.mzn]
    python3 cnf2mzn.py --batch sat.inst --out-dir data/sat_mzn/
"""

import argparse
import gzip
import lzma
import os
import sys
from pathlib import Path


def open_file_auto(path):
    path_str = str(path)
    if path_str.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    elif path_str.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        return open(path, "rt", encoding="utf-8", errors="replace")


def cnf_to_mzn(cnf_stream, out_stream):
    """
    Parses a DIMACS CNF stream and writes equivalent MiniZinc code to out_stream.
    """
    num_vars = 0
    num_clauses = 0
    header_written = False
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
            header_written = True
            continue

        tokens = line.split()
        for tok in tokens:
            lit = int(tok)
            if lit == 0:
                if current_clause:
                    # Write MiniZinc disjunction
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


def convert_single_file(input_path, output_path):
    input_path = Path(input_path)
    if output_path is None:
        name = input_path.stem
        if name.endswith(".cnf"):
            name = Path(name).stem
        output_path = input_path.parent / f"{name}.mzn"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_file_auto(input_path) as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        cnf_to_mzn(f_in, f_out)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert DIMACS CNF to MiniZinc (.mzn)")
    parser.add_argument("input", nargs="?", help="Input CNF file (.cnf, .cnf.xz, .cnf.gz)")
    parser.add_argument("-o", "--output", help="Output .mzn file path")
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    out_file = convert_single_file(args.input, args.output)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
