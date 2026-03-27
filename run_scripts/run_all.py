import argparse, subprocess


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=str, choices=['as', 'par'], required=True)
parser.add_argument('-f', '--features', type=str, choices=['wlc-0', 'wlc-1', 'wlc-2', 'wl-0', 'wl-1', 'wl-2', 'wln-0', 'wln-1', 'wln-2', 'wle-0', 'wle-1', 'wle-2', 'wlne-0', 'wlne-1', 'wlne-2', 'fzn2feat'], required=True)
parser.add_argument('-m', '--model', type=str, choices=['svc', 'rnd-forest', 'nn', 'f-knn', 'knn'], required=True)
parser.add_argument('-b', '--base-name', required=True)

args = parser.parse_args()

rnd_seeds = [12, 42, 55, 74, 83]#, 99, 123, 169, 200, 250]

if args.experiment == 'as':
    command = f'python src/algorithm_selection/algorithm_selection.py -f {args.features} -m {args.model}'
    for seed in rnd_seeds:
        result:str = args.base_name
        assert '{seed}' in result, '{cv} is required for as base-name'
        assert '{model}' in result, '{model} is required for as base-name'
        assert '{features}' in result, '{features} is required for as base-name'
        result = result.replace('{seed}',str(seed)).replace('{model}', args.model).replace('{features}', args.features)
        actual_command = command + f' --rnd-state {seed}' + f' --result {result}'
        print(actual_command)
        subprocess.run([actual_command], shell=True)

elif args.experiment == 'par':
    assert args.rnd_state is not None, '-r/--rnd-state is required for experiment par'
    command = f'python src/parallelise/parallelise.py -f {args.features} -m {args.model}'
    for rnd in args.rnd_state:
        result:str = args.base_name
        assert '{rnd-state}' in result, '{rnd-state} is required for par base-name'
        assert '{model}' in result, '{model} is required for par base-name'
        assert '{features}' in result, '{features} is required for par base-name'
        result = result.replace('{rnd-state}',str(rnd)).replace('{model}', args.model).replace('{features}', args.features)
        actual_command = command + f' -r {rnd}' + f' --result {result}'
        subprocess.run([actual_command], shell=True)

else:
    print(f'unknown experiment {args.experiment}')
