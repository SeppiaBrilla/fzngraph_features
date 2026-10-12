import os
import pandas as pd
import json
from tqdm import tqdm
import zerofolio as zf
from google import genai
from dotenv import load_dotenv, dotenv_values 

DATASET_CSV = "data/sat/sat_dataset.csv"
OUTPUT_CSV = "data/sat/sat_encoder.csv"

def embed(client, instance:str):
    text = zf.serialize(instance, strategy="line_shuffle", max_chars=10000, seed=42)
    res = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return res.embeddings[0].values

features = []
load_dotenv() 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

df = pd.read_csv(DATASET_CSV)
items = [(r['flatzinc'], r['instanceid']) for _, r in df.iterrows()]

for flatzinc_path, instanceid in tqdm(items):
    cnf_path = flatzinc_path.replace("fzn", "cnf") + ".xz"
    feats = embed(client, cnf_path)
    instance_dict = {}
    instance_dict["instanceid"] = instanceid
    instance_dict["flatzinc"] = flatzinc_path
    for i, f in enumerate(feats):
        instance_dict[i] = f
    features.append(instance_dict)


df = pd.DataFrame(features)
df.to_csv(OUTPUT_CSV, index=False)
