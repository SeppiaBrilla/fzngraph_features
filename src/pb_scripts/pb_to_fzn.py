import glob
import subprocess
from opb_converter import opb_to_mzn_content
from tqdm import tqdm
import lzma
import os
import re

files = glob.glob('data/selected-PB25/**/*.opb.xz', recursive=True) + glob.glob('data/selected-PB25/**/*.opb', recursive=True)
CACHE_FILE = '.cache/model.mzn'

for file in tqdm(files):
    if ".xz" in file:
        assert ".opb.xz" in file, file
        fzn_name = file.replace(".opb.xz", ".fzn")
    else:
        assert ".opb" in file, file
        fzn_name = file.replace(".opb", ".fzn")
    if os.path.exists(fzn_name):
        subprocess.run([f'rm {fzn_name}'], shell=True)

    try:
        with (lzma.open(file, 'rt', errors='ignore') if file.endswith('.xz') else open(file, 'r', errors='ignore')) as fp:
            first_line = fp.readline()
    except Exception:
        continue
    m = re.search(r'#variable=\s*(\d+)\s+#constraint=\s*(\d+)', first_line)
    if not m:
        continue
    v, c = int(m.group(1)), int(m.group(2))
    if v > 5000:
        continue
    mzn = opb_to_mzn_content(file)
    assert mzn is not None
    with open(CACHE_FILE, 'w') as f:
        f.write(mzn)
    subprocess.run([f'minizinc -c {CACHE_FILE} --solver gecode --no-output-ozn --fzn {fzn_name}'], shell=True)
    subprocess.run([f'rm {CACHE_FILE}'], shell=True)
    if not os.path.exists(fzn_name):
        print("WARNING: fzn", fzn_name, "missing")
