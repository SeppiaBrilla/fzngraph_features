import os, subprocess, uuid, tqdm, time

def make_flat(model:str, instance:str|None) -> str:
    id = '.cache/' + str(uuid.uuid4()) + '.fzn'
    subprocess.run([f'minizinc -c {model} {instance if instance else ""} --no-output-ozn --fzn {id} --solver gecode'], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return id

import socket
import time

def send_to_server(socket_path, args):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(socket_path)
    msg = "\0".join(args) + "\0\n"
    s.sendall(msg.encode())
    while True:
        data = s.recv(4096)
        if not data:
            break
    s.close()

def get_time(model:str, instance:str|None) -> tuple[float, float, float]:
    id = make_flat(model, instance)
    start_time = time.time()
    subprocess.run([f'/home/alessio/Documents/projects/mzn2feat/bin/fzn2feat {id}'], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end_time = time.time()
    fzn2feat_time = end_time - start_time
    
    start_time = time.time()
    send_to_server("/tmp/zinctowl_8.sock", [id, "-k", "1", "-m", "wl-nc", "--colors", "col.bin", "-t", "false", "-c", "8"])
    end_time = time.time()
    ZincToWl_8_time = end_time - start_time

    start_time = time.time()
    send_to_server("/tmp/zinctowl_8.sock", [id, "-k", "1", "-m", "wl-nc", "--colors", "col.bin", "-t", "false", "-c", "1"])
    end_time = time.time()
    ZincToWl_1_time = end_time - start_time

    start_time = time.time()
    subprocess.run([f'rm {id}'], shell=True)
    return fzn2feat_time, ZincToWl_8_time, ZincToWl_1_time

# Start servers automatically
env_8 = dict(os.environ, JULIA_NUM_THREADS="24")
print("Spawning ZincToWl servers...")
server_8 = subprocess.Popen(["ZincToWl", "--server", "/tmp/zinctowl_8.sock"], env=env_8, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2) # Give servers time to boot

pbar = tqdm.tqdm(total=1500)
results = []
for year in os.listdir("../ucloudExecutor/mzn-challenge/"):
    for problem in os.listdir("../ucloudExecutor/mzn-challenge/" + year):
        base_name = f'{problem}-{year}'
        folder = "../ucloudExecutor/mzn-challenge/" + year + "/" + problem
        if not os.path.isdir(folder):
            continue
        models = [file for file in os.listdir(folder) if file.endswith(".mzn")]
        instances = [file for file in os.listdir(folder) if file.endswith(".dzn") or file.endswith(".json")]
        assert len(models) >= 1, f"need at least one model ({folder})"
        if len(models) > 1:
            assert len(instances) == 0, f"if there is more than a model then we cannot have instances ({folder})"
            for model in models:
                name = f"{base_name}-{model.replace('.mzn', '')}.graph"
                res = get_time(f"{folder}/{model}", None)
                results.append({
                    "year": year,
                    "problem": problem,
                    "model": model,
                    "instance": None,
                    "fzn2feat_time": res[0],
                    "ZincToWl_8_time": res[1],
                    "ZincToWl_1_time": res[2],
                })
                pbar.update(1)
        else:
            assert len(instances) >= 1, f"one model requires many instances ({folder})"
            model = models[0]
            for instance in instances:
                name = f"{base_name}-{instance.replace('.dzn', '').replace('.json', '')}.graph"
                res = get_time(f"{folder}/{model}", f"{folder}/{instance}")
                results.append({
                    "year": year,
                    "problem": problem,
                    "model": model,
                    "instance": instance,
                    "fzn2feat_time": res[0],
                    "ZincToWl_8_time": res[1],
                    "ZincToWl_1_time": res[2],
                })
                pbar.update(1)

import pandas as pd

pd.DataFrame(results).to_csv("data/feature_extraction_costs.csv", index=False)

# Teardown servers
server_8.terminate()
try:
    os.remove("/tmp/zinctowl_8.sock")
except:
    pass