python run_scripts/run_all.py -e as -f "$1" -m svc -b data/as-scores-results/svc-scores/as_{features}-{model}-{cv}-{seed}.json
python run_scripts/run_all.py -e as -f "$1" -m rnd-forest -b data/as-scores-results/forest-scores/as_{features}-{model}-{cv}-{seed}.json
python run_scripts/run_all.py -e as -f "$1" -m nn -b data/as-scores-results/nn-scores/as_{features}-{model}-{cv}-{seed}.json

python run_scripts/run_all.py -e as-acc -f "$1" -m svc -b data/as-accuracy-results/svc-scores/as-accuracy_{features}-{model}-{cv}-{seed}.json
python run_scripts/run_all.py -e as-acc -f "$1" -m rnd-forest -b data/as-accuracy-results/forest-scores/as-accuracy_{features}-{model}-{cv}-{seed}.json
python run_scripts/run_all.py -e as-acc -f "$1" -m nn -b data/as-accuracy-results/nn-scores/as-accuracy_{features}-{model}-{cv}-{seed}.json

python run_scripts/run_all.py -e par -f "$1" -m svc -b data/parallelise-results/svc-parallelise/par_{features}-{model}-{cv}-{seed}.json
python run_scripts/run_all.py -e par -f "$1" -m rnd-forest -b data/parallelise-results/forest-parallelise/par_{features}-{model}-{cv}-{seed}.json
python run_scripts/run_all.py -e par -f "$1" -m nn -b data/parallelise-results/nn-parallelise/par_{features}-{model}-{cv}-{seed}.json
