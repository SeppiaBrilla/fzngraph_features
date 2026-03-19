# mkdir data/2fold-svc

PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-1 -m svc --cv-fold 0 --result data/2fold-svc/as_wl-1-svc-0.json --max-cv 2
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-1 -m svc --cv-fold 1 --result data/2fold-svc/as_wl-1-svc-1.json --max-cv 2

PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-2 -m svc --cv-fold 0 --result data/2fold-svc/as_wl-2-svc-0.json --max-cv 2
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-2 -m svc --cv-fold 1 --result data/2fold-svc/as_wl-2-svc-1.json --max-cv 2
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-1 -m svc --cv-fold 0 --result data/2fold-svc/as_wlne-1-svc-0.json --max-cv 2
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-1 -m svc --cv-fold 1 --result data/2fold-svc/as_wlne-1-svc-1.json --max-cv 2
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-2 -m svc --cv-fold 0 --result data/2fold-svc/as_wlne-2-svc-0.json --max-cv 2
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-2 -m svc --cv-fold 1 --result data/2fold-svc/as_wlne-2-svc-1.json --max-cv 2

#
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wln-2 -m svc --cv-fold 0 --result data/2fold-svc/as_wln-2-svc-0.json --max-cv 2
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wln-2 -m svc --cv-fold 1 --result data/2fold-svc/as_wln-2-svc-1.json --max-cv 2

# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m svc --cv-fold 0 --result data/2fold-svc/as_wlnc-1-svc-0.json --max-cv 2
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m svc --cv-fold 1 --result data/2fold-svc/as_wlnc-1-svc-1.json --max-cv 2
#
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m svc --cv-fold 0 --result data/2fold-svc/as_wlnc-2-svc-0.json --max-cv 2
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m svc --cv-fold 1 --result data/2fold-svc/as_wlnc-2-svc-1.json --max-cv 2

# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f fzn2feat -m svc --cv-fold 0 --result data/2fold-svc/as_fzn2feat-svc-0.json --max-cv 2
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f fzn2feat -m svc --cv-fold 1 --result data/2fold-svc/as_fzn2feat-svc-1.json --max-cv 2
#
# mkdir data/3fold-svc
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-1 -m svc --cv-fold 0 --result data/3fold-svc/as_wl-1-svc-0.json --max-cv 3
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-1 -m svc --cv-fold 1 --result data/3fold-svc/as_wl-1-svc-1.json --max-cv 3
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-1 -m svc --cv-fold 2 --result data/3fold-svc/as_wl-1-svc-2.json --max-cv 3

PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-2 -m svc --cv-fold 0 --result data/3fold-svc/as_wl-2-svc-0.json --max-cv 3
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-2 -m svc --cv-fold 1 --result data/3fold-svc/as_wl-2-svc-1.json --max-cv 3
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-2 -m svc --cv-fold 2 --result data/3fold-svc/as_wl-2-svc-2.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-1 -m svc --cv-fold 0 --result data/3fold-svc/as_wlne-1-svc-0.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-1 -m svc --cv-fold 1 --result data/3fold-svc/as_wlne-1-svc-1.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-1 -m svc --cv-fold 2 --result data/3fold-svc/as_wlne-1-svc-2.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-2 -m svc --cv-fold 0 --result data/3fold-svc/as_wlne-2-svc-0.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-2 -m svc --cv-fold 1 --result data/3fold-svc/as_wlne-2-svc-1.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-2 -m svc --cv-fold 2 --result data/3fold-svc/as_wlne-2-svc-2.json --max-cv 3

#
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wln-2 -m svc --cv-fold 0 --result data/3fold-svc/as_wln-2-svc-0.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wln-2 -m svc --cv-fold 1 --result data/3fold-svc/as_wln-2-svc-1.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wln-2 -m svc --cv-fold 2 --result data/3fold-svc/as_wln-2-svc-2.json --max-cv 3

# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m svc --cv-fold 0 --result data/3fold-svc/as_wlnc-1-svc-0.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m svc --cv-fold 1 --result data/3fold-svc/as_wlnc-1-svc-1.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m svc --cv-fold 2 --result data/3fold-svc/as_wlnc-1-svc-2.json --max-cv 3
#
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m svc --cv-fold 0 --result data/3fold-svc/as_wlnc-2-svc-0.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m svc --cv-fold 1 --result data/3fold-svc/as_wlnc-2-svc-1.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m svc --cv-fold 2 --result data/3fold-svc/as_wlnc-2-svc-2.json --max-cv 3

# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f fzn2feat -m svc --cv-fold 0 --result data/3fold-svc/as_fzn2feat-svc-0.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f fzn2feat -m svc --cv-fold 1 --result data/3fold-svc/as_fzn2feat-svc-1.json --max-cv 3
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f fzn2feat -m svc --cv-fold 2 --result data/3fold-svc/as_fzn2feat-svc-2.json --max-cv 3
#
# mkdir data/5fold-svc
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-1 -m svc --cv-fold 0 --result data/5fold-svc/as_wl-1-svc-0.json --max-cv 5
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-1 -m svc --cv-fold 1 --result data/5fold-svc/as_wl-1-svc-1.json --max-cv 5
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-1 -m svc --cv-fold 2 --result data/5fold-svc/as_wl-1-svc-2.json --max-cv 5
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-1 -m svc --cv-fold 3 --result data/5fold-svc/as_wl-1-svc-3.json --max-cv 5
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-1 -m svc --cv-fold 4 --result data/5fold-svc/as_wl-1-svc-4.json --max-cv 5

PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-2 -m svc --cv-fold 0 --result data/5fold-svc/as_wl-2-svc-0.json --max-cv 5
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-2 -m svc --cv-fold 1 --result data/5fold-svc/as_wl-2-svc-1.json --max-cv 5
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-2 -m svc --cv-fold 2 --result data/5fold-svc/as_wl-2-svc-2.json --max-cv 5
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-2 -m svc --cv-fold 3 --result data/5fold-svc/as_wl-2-svc-3.json --max-cv 5
PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wl-2 -m svc --cv-fold 4 --result data/5fold-svc/as_wl-2-svc-4.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-1 -m svc --cv-fold 0 --result data/5fold-svc/as_wlne-1-svc-0.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-1 -m svc --cv-fold 1 --result data/5fold-svc/as_wlne-1-svc-1.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-1 -m svc --cv-fold 2 --result data/5fold-svc/as_wlne-1-svc-2.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-1 -m svc --cv-fold 3 --result data/5fold-svc/as_wlne-1-svc-3.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-1 -m svc --cv-fold 4 --result data/5fold-svc/as_wlne-1-svc-4.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-2 -m svc --cv-fold 0 --result data/5fold-svc/as_wlne-2-svc-0.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-2 -m svc --cv-fold 1 --result data/5fold-svc/as_wlne-2-svc-1.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-2 -m svc --cv-fold 2 --result data/5fold-svc/as_wlne-2-svc-2.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-2 -m svc --cv-fold 3 --result data/5fold-svc/as_wlne-2-svc-3.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlne-2 -m svc --cv-fold 4 --result data/5fold-svc/as_wlne-2-svc-4.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wln-2 -m svc --cv-fold 0 --result data/5fold-svc/as_wln-2-svc-0.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wln-2 -m svc --cv-fold 1 --result data/5fold-svc/as_wln-2-svc-1.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wln-2 -m svc --cv-fold 2 --result data/5fold-svc/as_wln-2-svc-2.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wln-2 -m svc --cv-fold 3 --result data/5fold-svc/as_wln-2-svc-3.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wln-2 -m svc --cv-fold 4 --result data/5fold-svc/as_wln-2-svc-4.json --max-cv 5

# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m svc --cv-fold 0 --result data/5fold-svc/as_wlnc-1-svc-0.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m svc --cv-fold 1 --result data/5fold-svc/as_wlnc-1-svc-1.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m svc --cv-fold 2 --result data/5fold-svc/as_wlnc-1-svc-2.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m svc --cv-fold 3 --result data/5fold-svc/as_wlnc-1-svc-3.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-1 -m svc --cv-fold 4 --result data/5fold-svc/as_wlnc-1-svc-4.json --max-cv 5

# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m svc --cv-fold 0 --result data/5fold-svc/as_wlnc-2-svc-0.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m svc --cv-fold 1 --result data/5fold-svc/as_wlnc-2-svc-1.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m svc --cv-fold 2 --result data/5fold-svc/as_wlnc-2-svc-2.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m svc --cv-fold 3 --result data/5fold-svc/as_wlnc-2-svc-3.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f wlc-2 -m svc --cv-fold 4 --result data/5fold-svc/as_wlnc-2-svc-4.json --max-cv 5

# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f fzn2feat -m svc --cv-fold 0 --result data/5fold-svc/as_fzn2feat-svc-0.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f fzn2feat -m svc --cv-fold 1 --result data/5fold-svc/as_fzn2feat-svc-1.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f fzn2feat -m svc --cv-fold 2 --result data/5fold-svc/as_fzn2feat-svc-2.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f fzn2feat -m svc --cv-fold 3 --result data/5fold-svc/as_fzn2feat-svc-3.json --max-cv 5
# PYTHONHASHSEED=42 python src/algorithm_selection/algorithm_selection.py -f fzn2feat -m svc --cv-fold 4 --result data/5fold-svc/as_fzn2feat-svc-4.json --max-cv 5
