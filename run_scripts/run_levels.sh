python run_scripts/run_all.py -e as -f "$1" -m svc -b data/as-scores-results/svc-scores/as_{features}_alllevels-{model}-{cv}-{seed}.json --all-levels
python run_scripts/run_all.py -e as -f "$1" -m rnd-forest -b data/as-scores-results/forest-scores/as_{features}_alllevels-{model}-{cv}-{seed}.json --all-levels

python run_scripts/run_all.py -e as-acc -f "$1" -m svc -b data/as-accuracy-results/svc-scores/as-accuracy_{features}_alllevels-{model}-{cv}-{seed}.json --all-levels
python run_scripts/run_all.py -e as-acc -f "$1" -m rnd-forest -b data/as-accuracy-results/forest-scores/as-accuracy_{features}_alllevels-{model}-{cv}-{seed}.json --all-levels

python run_scripts/run_all.py -e par -f "$1" -m svc -b data/parallelise-results/svc-scores/par_{features}_alllevels-{model}-{cv}-{seed}.json --all-levels
python run_scripts/run_all.py -e par -f "$1" -m rnd-forest -b data/parallelise-results/forest-scores/par_{features}_alllevels-{model}-{cv}-{seed}.json --all-levels