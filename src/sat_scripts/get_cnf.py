import os, subprocess, tqdm

max_ram_bytes = int(10 * 1024 * 1024 * 1024)
pbar = tqdm.tqdm(total=1500)
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
                name = f"data/cnfs/{base_name}-{model.replace('.mzn', '')}.cnf"
                if not os.path.exists(name):
                    subprocess.run([f"minizinc -c {folder}/{model} --no-output-ozn --fzn ./.cache/flat.fzn --solver picat"], shell=True, stdout=subprocess.DEVNULL)
                    subprocess.run([f'prlimit --as={max_ram_bytes} picat ~/.local/opt/fzn_picat/fzn_picat_sat ./.cache/flat.fzn dumpcnf'], shell=True, stdout=subprocess.DEVNULL)
                    if os.path.exists("__tmp.cnf"):
                        os.rename("__tmp.cnf", name)
                    else:
                        print(f"Failed to create CNF for {folder}/{model} {folder}/{instance}")
                pbar.update(1)
        else:
            assert len(instances) >= 1, f"one model requires many instances ({folder})"
            model = models[0]
            for instance in instances:
                name = f"data/cnfs/{base_name}-{instance.replace('.dzn', '').replace('.json', '')}.cnf"
                if not os.path.exists(name):
                    subprocess.run([f"minizinc -c {folder}/{model} {folder}/{instance} --no-output-ozn --fzn ./.cache/flat.fzn --solver picat"], shell=True, stdout=subprocess.DEVNULL)
                    subprocess.run([f'prlimit --as={max_ram_bytes} picat ~/.local/opt/fzn_picat/fzn_picat_sat ./.cache/flat.fzn dumpcnf'], shell=True, stdout=subprocess.DEVNULL)
                    if os.path.exists("__tmp.cnf"):
                        os.rename("__tmp.cnf", name)
                    else:
                        print(f"Failed to create CNF for {folder}/{model} {folder}/{instance}")
                pbar.update(1)
#picat fzn_picat_sat ./test.fzn dumpcnf