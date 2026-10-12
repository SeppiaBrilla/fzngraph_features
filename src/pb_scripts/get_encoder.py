import glob
import os
import pandas as pd
import subprocess
from tqdm import tqdm
import zerofolio as zf
from google import genai
from dotenv import load_dotenv, dotenv_values 

def embed(client, instance:str):
    text = zf.serialize(instance, strategy="line_shuffle", max_chars=10000, seed=42)
    res = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return res.embeddings[0].values

load_dotenv() 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

files = glob.glob('data/selected-PB25/**/*.opb.xz', recursive=True) + glob.glob('data/selected-PB25/**/*.opb', recursive=True)

instances = []
for file in tqdm(files):
    if file.endswith(".opb.xz"):
        flat = file.replace(".opb.xz", ".fzn")
        #extract to opb
        subprocess.run([f"unxz -k {file}"], shell=True)
        file = file.replace(".opb.xz", ".opb")
    else:
        flat = file.replace(".opb", ".fzn")
    if not os.path.exists(flat):
        continue

    inst = {
        'flatzinc': flat,
    }
    feats = embed(client, file)
    for i, feat in enumerate(feats):
        inst[i] = feat

    instances.append(inst)

df = pd.DataFrame(instances)
df.to_csv("data/pb_encoder.csv", index=None)


# df = pd.read_csv(DATASET_CSV)
# items = [(r['flatzinc'], r['instanceid']) for _, r in df.iterrows()]

# for flatzinc_path, instanceid in tqdm(items):
#     cnf_path = flatzinc_path.replace("fzn", "cnf") + ".xz"
#     feats = embed(client, cnf_path)
#     instance_dict = {}
#     instance_dict["instanceid"] = instanceid
#     instance_dict["flatzinc"] = flatzinc_path
#     for i, f in enumerate(feats):
#         instance_dict[i] = f
#     features.append(instance_dict)


# df = pd.DataFrame(features)
# df.to_csv(OUTPUT_CSV, index=False)