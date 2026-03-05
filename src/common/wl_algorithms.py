from typing import Literal
from .graph_loader import Graph

COLORS = {}

def standard_wl(graph:Graph, colors:dict, max_iter:int=10, training:bool=True, max_colors:int|None=None, with_neighbours:bool=False) -> list[int]|list[list[int]]:
    """
    implements the standard wl algorithm wihout taking into account neither node types nor edge types
    """

    node_colors:list[str] = ['1' for _ in graph.nodes]
    node_idx = {node.label:idx for idx, node in enumerate(graph.nodes)}

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

    if with_neighbours:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])
        return [
            [int(node_colors[i])] + [int(c) for c in sorted(neighbour_colors[i])] for i in range(len(neighbour_colors))
        ]

    return [int(c) for c in node_colors]

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

    if with_neighbours:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])
        return [
            [int(node_colors[i])] + [int(c) for c in sorted(neighbour_colors[i])] for i in range(len(neighbour_colors))
        ]

    # if not training:
    #     print(out)
    #     print('===================')
    return [int(c) for c in node_colors]

def wl_with_edge_features(graph:Graph, colors:dict, max_iter:int=10, training:bool=True, max_colors:int|None=None, with_neighbours:bool=False) -> list[int]|list[list[int]]:
    """
    implements thewl algorithm taking into account edge types as features
    """

    node_colors:list[str] = ['1' for _ in graph.nodes]
    node_idx = {node.label:idx for idx, node in enumerate(graph.nodes)}

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

    if with_neighbours:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])
        return [
            [int(node_colors[i])] + [int(c) for c in sorted(neighbour_colors[i])] for i in range(len(neighbour_colors))
        ]

    return [int(c) for c in node_colors]

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

    if with_neighbours:
        neighbour_colors = [[] for _ in node_colors]
        for (_from, _to), _ in graph.edge_iterator:
            from_idx = node_idx[_from.label]
            to_idx = node_idx[_to.label]
            neighbour_colors[to_idx].append(node_colors[from_idx])
        return [
            [int(node_colors[i])] + [int(c) for c in sorted(neighbour_colors[i])] for i in range(len(neighbour_colors))
        ]

    return [int(c) for c in node_colors]

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
    from graph_loader import load_grap
    with open('graphs/accap-sep-accap_a3_f20_t10.graph') as f:
        graph = load_grap(f)
    n_iterations = 3
    colors = {}
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
