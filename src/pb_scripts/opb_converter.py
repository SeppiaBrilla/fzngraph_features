import sys
import lzma
import re

def opb_to_mzn_content(filepath):
    lines = []
    if filepath.endswith('.xz'):
        with lzma.open(filepath, 'rt', errors='ignore') as fp:
            for line in fp:
                line = line.strip()
                if line and not line.startswith('*'):
                    lines.append(line)
    else:
        with open(filepath, 'r', errors='ignore') as fp:
            for line in fp:
                line = line.strip()
                if line and not line.startswith('*'):
                    lines.append(line)

    var_indices = set()
    obj_tokens = []
    constrs = []

    for line in lines:
        if line.startswith('min:'):
            obj_tokens = line[4:].rstrip(';').strip().split()
        else:
            constrs.append(line.rstrip(';').strip())

    for t in re.findall(r'x(\d+)', ' '.join(lines)):
        var_indices.add(int(t))

    if not var_indices:
        return None
    max_var = max(var_indices)

    mzn_lines = []
    mzn_lines.append(f'array[1..{max_var}] of var 0..1: x;')

    def parse_expr(tokens):
        terms = []
        i = 0
        while i < len(tokens):
            coeff_str = tokens[i]
            coeff = int(coeff_str)
            i += 1
            lits = []
            while i < len(tokens) and not (tokens[i] in ['>=', '=', '<='] or (tokens[i][0] in '+-' and len(tokens[i]) > 1 and tokens[i][1:].isdigit())):
                lits.append(tokens[i])
                i += 1

            lit_exprs = []
            for lit in lits:
                if lit.startswith('~x'):
                    idx = int(lit[2:])
                    lit_exprs.append(f'(1 - x[{idx}])')
                elif lit.startswith('x'):
                    idx = int(lit[1:])
                    lit_exprs.append(f'x[{idx}]')

            if not lit_exprs:
                terms.append(f'{coeff}')
            else:
                prod = ' * '.join(lit_exprs)
                if coeff == 1:
                    terms.append(f'{prod}')
                elif coeff == -1:
                    terms.append(f'-{prod}')
                else:
                    terms.append(f'{coeff} * {prod}')
        return ' + '.join(terms) if terms else '0'

    for c in constrs:
        m = re.search(r'([>=<]+)\s*(-?\d+)$', c)
        if not m:
            continue
        op = m.group(1)
        rhs = m.group(2)
        lhs_tokens = c[:m.start()].strip().split()
        lhs_expr = parse_expr(lhs_tokens)
        if op == '=':
            mzn_lines.append(f'constraint {lhs_expr} == {rhs};')
        else:
            mzn_lines.append(f'constraint {lhs_expr} {op} {rhs};')

    if obj_tokens:
        obj_expr = parse_expr(obj_tokens)
        mzn_lines.append(f'var int: obj = {obj_expr};')
        mzn_lines.append('solve minimize obj;')
    else:
        mzn_lines.append('solve satisfy;')

    return '\n'.join(mzn_lines)


if __name__ == '__main__':
    print(opb_to_mzn_content(sys.argv[1]))
