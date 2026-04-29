from typing import Literal
try:
    from .graph_loader import is_global
except:
    from graph_loader import is_global
try:
    from .graph_loader import Graph
except:
    from graph_loader import Graph

COLORS = {}

def standard_wl(graph:Graph, colors:dict, max_iter:int=10, training:bool=True, max_colors:int|None=None, with_neighbours:bool=False) -> list[int]|list[list[int]]:
    """
    implements the standard wl algorithm wihout taking into account neither node types nor edge types
    """

    node_colors:list[str] = ['1' for _ in graph.nodes]
    node_idx = {node.label:idx for idx, node in enumerate(graph.nodes)}

    levels = [[int(c) for c in node_colors]]

    changed = True
    iter = 0
    while changed and iter < max_iter:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])

        updated_colors = []
        for i in range(len(neighbour_colors)):
            updated_colors.append(node_colors[i] + "".join(sorted(neighbour_colors[i])))

        if training:
            if max_colors == None or len(colors.keys()) <= max_colors:
                for uc in sorted(set(updated_colors)):
                    if not uc in colors:
                        colors[uc] = str(hash(uc))
        new_node_colors = [colors[uc] if uc in colors else node_colors[i] for i, uc in enumerate(updated_colors)]
        iter += 1
        changed = not node_colors == new_node_colors
        node_colors = new_node_colors
        levels.append([int(c) for c in node_colors])

    if with_neighbours:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])
        levels[-1] = [
            [int(node_colors[i])] + [int(c) for c in sorted(neighbour_colors[i])] for i in range(len(neighbour_colors))
        ]
        return levels

    return levels

def wl_with_node_features(graph:Graph, colors:dict, max_iter:int=10, training:bool=True, max_colors:int|None=None, with_neighbours:bool=False) -> list[int]|list[list[int]]:
    """
    implements thewl algorithm taking into account node types as features
    """

    node_colors:list[str] = [str(node._type) for node in graph.nodes]
    node_idx = {node.label:idx for idx, node in enumerate(graph.nodes)}
    for uc in sorted(set(node_colors)):
        if not uc in colors:
            colors[uc] = str(hash(uc))
    node_colors = [colors[uc] for uc in node_colors]

    levels = [[int(c) for c in node_colors]]

    changed = True
    iter = 0
    # out = 0
    while changed and iter < max_iter:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])

        updated_colors = []
        for i in range(len(neighbour_colors)):
            updated_colors.append(node_colors[i] + "".join(sorted(neighbour_colors[i])))

        if training:
            if max_colors == None or len(colors.keys()) <= max_colors:
                for uc in sorted(set(updated_colors)):
                    if not uc in colors:
                        colors[uc] = str(hash(uc))

        # if not training:
        #     if any(not uc in colors for uc in updated_colors):
        #         out += sum([1 if not uc in colors else 0 for uc in updated_colors])
        new_node_colors = [colors[uc] if uc in colors else node_colors[i] for i, uc in enumerate(updated_colors)]
        iter += 1
        changed = not node_colors == new_node_colors
        node_colors = new_node_colors
        levels.append([int(c) for c in node_colors])

    if with_neighbours:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])
        levels[-1] = [
            [int(node_colors[i])] + [int(c) for c in sorted(neighbour_colors[i])] for i in range(len(neighbour_colors))
        ]
        return levels

    # if not training:
    #     print(out)
    #     print('===================')
    return levels

def wl_with_edge_features(graph:Graph, colors:dict, max_iter:int=10, training:bool=True, max_colors:int|None=None, with_neighbours:bool=False) -> list[int]|list[list[int]]:
    """
    implements thewl algorithm taking into account edge types as features
    """

    node_colors:list[str] = ['1' for _ in graph.nodes]
    node_idx = {node.label:idx for idx, node in enumerate(graph.nodes)}

    levels = [[int(c) for c in node_colors]]

    changed = True
    iter = 0
    while changed and iter < max_iter:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), e in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx] + ',' + e.label)

        updated_colors = []
        for i in range(len(neighbour_colors)):
            updated_colors.append(node_colors[i] + "".join(sorted(neighbour_colors[i])))

        if training:
            if max_colors == None or len(colors.keys()) <= max_colors:
                for uc in sorted(set(updated_colors)):
                    if not uc in colors:
                        colors[uc] = str(hash(uc))

        new_node_colors = [colors[uc] if uc in colors else node_colors[i] for i, uc in enumerate(updated_colors)]
        iter += 1
        changed = not node_colors == new_node_colors
        node_colors = new_node_colors
        levels.append([int(c) for c in node_colors])

    if with_neighbours:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])
        levels[-1] = [
            [int(node_colors[i])] + [int(c) for c in sorted(neighbour_colors[i])] for i in range(len(neighbour_colors))
        ]
        return levels

    return levels

def wl_with_node_and_edge_features(graph:Graph, colors:dict, max_iter:int=10, training:bool=True, max_colors:int|None=None, with_neighbours:bool=False) -> list[int]|list[list[int]]:
    """
    implements thewl algorithm taking into account node and edge types as features
    """

    node_colors:list[str] = [str(node._type) for node in graph.nodes]
    node_idx = {node.label:idx for idx, node in enumerate(graph.nodes)}
    for uc in sorted(set(node_colors)):
        if not uc in colors:
            colors[uc] = str(hash(uc))
    node_colors = [colors[uc] for uc in node_colors]

    levels = [[int(c) for c in node_colors]]

    changed = True
    iter = 0
    while changed and iter < max_iter:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), e in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx] + ',' + e.label)

        updated_colors = []
        for i in range(len(neighbour_colors)):
            updated_colors.append(node_colors[i] + "".join(sorted(neighbour_colors[i])))

        if training:
            if max_colors == None or len(colors.keys()) <= max_colors:
                for uc in sorted(set(updated_colors)):
                    if not uc in colors:
                        colors[uc] = str(hash(uc))

        new_node_colors = [colors[uc] if uc in colors else node_colors[i] for i, uc in enumerate(updated_colors)]
        iter += 1
        changed = not node_colors == new_node_colors
        node_colors = new_node_colors
        levels.append([int(c) for c in node_colors])

    if with_neighbours:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])
        levels[-1] = [
            [int(node_colors[i])] + [int(c) for c in sorted(neighbour_colors[i])] for i in range(len(neighbour_colors))
        ]
        return levels

    return levels

def wl_extended_features(graph:Graph, colors:dict, max_iter:int=1, training:bool=True) -> tuple[list[int],dict]:
    node_colors:list[str] = [str(node._type if node._type != 'literal_node' else 'par_node') for node in graph.nodes]
    node_idx = {node.label:idx for idx, node in enumerate(graph.nodes)}
    for uc in sorted(set(node_colors)):
        if not uc in colors:
            colors[uc] = str(hash(uc))
    node_colors = [colors[uc] for uc in node_colors]

    levels_list = [[int(c) for c in node_colors]]

    constraints_per_variable = 0
    constraints_per_par = 0
    n_var, n_par = 0, 0
    pairs = {}

    levels = {}

    for iter in range(max_iter):
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            if is_global(_to._type) or 'lin_' in _to._type or 'multi_' in _to._type:
                continue
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])

        updated_colors = []
        for i in range(len(neighbour_colors)):
            updated_colors.append(node_colors[i] + "".join(sorted(neighbour_colors[i])))

        if training:
            for uc in sorted(set(updated_colors)):
                if uc in colors.values():
                    continue
                if not uc in colors:
                    colors[uc] = str(hash(uc))

        new_node_colors = [colors[uc] if uc in colors else node_colors[i] for i, uc in enumerate(updated_colors)]
        levels[iter] = [int(c) for c in node_colors]
        node_colors = new_node_colors
        levels_list.append([int(c) for c in node_colors])

    for node in graph.nodes:
        if node._type == 'var_node':
            constraints_per_variable += len(graph.edge_from(node))
            n_var += 1
        elif node._type in ['par_node', 'literal_node']:
            constraints_per_par += len(graph.edge_from(node))
            n_par += 1
        elif is_global(node._type) or 'lin_' in node._type or 'multi_' in node._type:
            for _from, _ in graph.edge_to(node):
                pair = (_from._type if _from._type != 'literal_node' else 'par_node', node._type)
                if not pair in pairs:
                    pairs[pair] = 0
                pairs[pair] += 1

    globals_set = sorted(set(p[1] for p in pairs.keys()))
    for g in globals_set:
        color = g + ',' + ''.join(sorted(t for t, f in pairs.keys() if f == g))
        h = str(hash(g))
        assert h in node_colors, (g, pairs)
        if training and not color in colors:
            colors[color] = str(hash(color))
        if color in colors:
            node_colors[node_colors.index(h)] = str(hash(color))
    
    levels_list[-1] = [int(c) for c in node_colors]
    
    extra_info = {
        'levels': levels,
        'globals_pairs': pairs,
        'cpv': constraints_per_variable / n_var,
        'cpp': constraints_per_par / n_par,
        'n_nodes': len(graph.nodes)
    }
    return levels_list, extra_info


def wl_extended_features_with_edges(graph:Graph, colors:dict, max_iter:int=1, training:bool=True) -> tuple[list[int],dict]:
    node_colors:list[str] = [str(node._type if node._type != 'literal_node' else 'par_node') for node in graph.nodes]
    node_idx = {node.label:idx for idx, node in enumerate(graph.nodes)}
    for uc in sorted(set(node_colors)):
        if not uc in colors:
            colors[uc] = str(hash(uc))
    node_colors = [colors[uc] for uc in node_colors]

    levels_list = [[int(c) for c in node_colors]]

    constraints_per_variable = 0
    constraints_per_par = 0
    n_var, n_par = 0, 0
    pairs = {}

    levels = {}

    for iter in range(max_iter):
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), e in graph.edge_iterator:
            if is_global(_to._type) or 'lin_' in _to._type or 'multi_' in _to._type:
                continue
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx] + ',' + e.label)

        updated_colors = []
        for i in range(len(neighbour_colors)):
            updated_colors.append(node_colors[i] + "".join(sorted(neighbour_colors[i])))

        if training:
            for uc in sorted(set(updated_colors)):
                if uc in colors.values():
                    continue
                if not uc in colors:
                    colors[uc] = str(hash(uc))

        new_node_colors = [colors[uc] if uc in colors else node_colors[i] for i, uc in enumerate(updated_colors)]
        levels[iter] = [int(c) for c in node_colors]
        node_colors = new_node_colors
        levels_list.append([int(c) for c in node_colors])

    for node in graph.nodes:
        if node._type == 'var_node':
            constraints_per_variable += len(graph.edge_from(node))
            n_var += 1
        elif node._type in ['par_node', 'literal_node']:
            constraints_per_par += len(graph.edge_from(node))
            n_par += 1
        elif is_global(node._type) or 'lin_' in node._type or 'multi_' in node._type:
            for _from, _ in graph.edge_to(node):
                pair = (_from._type if _from._type != 'literal_node' else 'par_node', node._type)
                if not pair in pairs:
                    pairs[pair] = 0
                pairs[pair] += 1

    globals_set = sorted(set(p[1] for p in pairs.keys()))
    for g in globals_set:
        color = g + ',' + ''.join(sorted(t for t, f in pairs.keys() if f == g))
        h = str(hash(g))
        assert h in node_colors, (g, pairs)
        if training and not color in colors:
            colors[color] = str(hash(color))
        if color in colors:
            node_colors[node_colors.index(h)] = str(hash(color))
            
    levels_list[-1] = [int(c) for c in node_colors]
    
    extra_info = {
        'levels': levels,
        'globals_pairs': pairs,
        'cpv': constraints_per_variable / n_var,
        'cpp': constraints_per_par / n_par,
        'n_nodes': len(graph.nodes)
    }
    return levels_list, extra_info


def wl_features(graph:Graph,
                colors:dict,
                max_iter:int=10,
                training:bool=True,
                wl_type:Literal['standard','node_features','edge_features','node_edge_features']='standard',
                max_colors:int|None=None,
                with_neighbours:bool=False) -> list[int]|list[list[int]]:
    if wl_type == 'standard':
        return standard_wl(graph, colors, max_iter, training, max_colors, with_neighbours)
    elif wl_type == 'edge_features':
        return wl_with_edge_features(graph, colors, max_iter, training, max_colors, with_neighbours)
    elif wl_type == 'node_features':
        return wl_with_node_features(graph, colors, max_iter, training, max_colors, with_neighbours)
    elif wl_type == 'node_edge_features':
        return wl_with_node_and_edge_features(graph, colors, max_iter, training, max_colors, with_neighbours)

    raise Exception(f'unrecognised wl_type: {wl_type}')

if __name__ == '__main__':
    from graph_loader import load_graph
    # with open('./data/graphs/tower-sep-tower_070_070_15_070-08.graph') as f:
    with open('./data/graphs/model4_opt-sep-test05.graph') as f:
        graph = load_graph(f)
    n_iterations = 3
    colors = {}
    wl_features(graph, colors, max_iter=n_iterations, wl_type='standard')
    print(f'number of colors with standard wl ({n_iterations} iters):', len(colors.keys()))
    colors = {}
    wl_features(graph, colors, max_iter=n_iterations, wl_type='node_features')
    print(f'number of colors with node-features agumented wl ({n_iterations} iters):', len(colors.keys()))
    colors = {}
    wl_features(graph, colors, max_iter=n_iterations, wl_type='edge_features')
    print(f'number of colors with edge-features agumented wl ({n_iterations} iters):', len(colors.keys()))
    colors = {}
    wl_features(graph, colors, max_iter=n_iterations, wl_type='node_edge_features')
    print(f'number of colors with node and edge-features agumented wl ({n_iterations} iters):', len(colors.keys()))
    print("===========================================================================")
    colors = {}
    wl_extended_features(graph, colors)
    print(len(colors))
