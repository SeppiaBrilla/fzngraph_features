import argparse, subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=str, choices=['as', 'par'], required=True)
parser.add_argument('-f', '--features', type=str, choices=['wlc-0', 'wlc-1', 'wlc-2', 'wl-0', 'wl-1', 'wl-2', 'wln-0', 'wln-1', 'wln-2', 'wle-0', 'wle-1', 'wle-2', 'wlne-0', 'wlne-1', 'wlne-2', 'fzn2feat'], required=True)
parser.add_argument('-m', '--model', type=str, choices=['svc', 'rnd-forest', 'nn', 'f-knn', 'knn'], required=True)
parser.add_argument('-b', '--base-name', required=True)

args = parser.parse_args()

max_cv = 5
rnd_seeds = [7, 12, 42]

if args.experiment == 'as':
    command = f'PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f {args.features} -m {args.model}'
    for seed in rnd_seeds:
        for cv in range(max_cv):
            result:str = args.base_name
            assert '{seed}' in result, '{seed} is required for as base-name'
            assert '{cv}' in result, '{cv} is required for as base-name'
            assert '{model}' in result, '{model} is required for as base-name'
            assert '{features}' in result, '{features} is required for as base-name'
            result = result.replace('{seed}',str(seed)).replace('{cv}',str(cv)).replace('{model}', args.model).replace('{features}', args.features)
            actual_command = command + f' --rnd-state {seed} --max-cv {max_cv} --cv-fold {cv} --result {result}'
            print(actual_command)
            subprocess.run([actual_command], shell=True)

elif args.experiment == 'par':
    command = f'PYTHONHASHSEED=42 python src/parallelise/parallelise.py -f {args.features} -m {args.model}'
    for rnd in rnd_seeds:
        result:str = args.base_name
        assert '{seed}' in result, '{seed} is required for par base-name'
        assert '{model}' in result, '{model} is required for par base-name'
        assert '{features}' in result, '{features} is required for par base-name'
        result = result.replace('{seed}',str(rnd)).replace('{model}', args.model).replace('{features}', args.features)
        actual_command = command + f' -r {rnd} --result {result}'
        print(actual_command)
        subprocess.run([actual_command], shell=True)

else:
    print(f'unknown experiment {args.experiment}')
