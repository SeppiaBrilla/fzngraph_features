import argparse, subprocess, os

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=str, choices=['as', 'as-acc', 'par'], required=True)
parser.add_argument('-f', '--features', type=str, choices=['wlce-1', 'wlce-2', 'wlc-0', 'wlc-1', 'wlc-2', 'wlcc', 'wl-0', 'wl-1', 'wl-2', 'wln-0', 'wln-1', 'wln-2', 'wle-0', 'wle-1', 'wle-2', 'wlne-0', 'wlne-1', 'wlne-2', 'fzn2feat'], required=True)
parser.add_argument('-m', '--model', type=str, choices=['svc', 'rnd-forest', 'nn', 'nn-torch', 'f-knn', 'knn'], required=True)
parser.add_argument('-b', '--base-name', required=True)
parser.add_argument('--all-levels', action='store_true', help='use all WL levels')

args = parser.parse_args()

max_cv = 5
rnd_seeds = [7, 12, 42, 72, 123, 156, 197, 205, 224, 242]

if args.experiment == 'as':
    command = f'PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f {args.features} -m {args.model}'
    if args.all_levels:
        command += ' --all-levels'
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
            if not os.path.exists(result):
                subprocess.run([actual_command], shell=True)

elif args.experiment == 'as-acc':
    command = f'PYTHONHASHSEED=42 python src/algorithm_selection_accuracy/algorithm_selection_accuracy.py -f {args.features} -m {args.model}'
    if args.all_levels:
        command += ' --all-levels'
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
            if not os.path.exists(result):
                subprocess.run([actual_command], shell=True)

elif args.experiment == 'par':
    command = f'PYTHONHASHSEED=42 python src/parallelise/parallelise.py -f {args.features} -m {args.model}'
    if args.all_levels:
        command += ' --all-levels'
    for seed in rnd_seeds:
        for cv in range(max_cv):
            result:str = args.base_name
            assert '{seed}' in result, '{seed} is required for par base-name'
            assert '{cv}' in result, '{cv} is required for par base-name'
            assert '{model}' in result, '{model} is required for par base-name'
            assert '{features}' in result, '{features} is required for par base-name'
            result = result.replace('{seed}',str(seed)).replace('{cv}',str(cv)).replace('{model}', args.model).replace('{features}', args.features)
            actual_command = command + f' --rnd-state {seed} --max-cv {max_cv} --cv-fold {cv} --result {result}'
            print(actual_command)
            if not os.path.exists(result):
                subprocess.run([actual_command], shell=True)

else:
    print(f'unknown experiment {args.experiment}')
