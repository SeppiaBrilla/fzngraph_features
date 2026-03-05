from typing import TextIO, Literal, Any, Iterable
import networkx as nx

NODE_TYPE = Literal['var_node', 'par_node', 'literal_node', 'mult_node', 'sum_node', 'equality_node', 'inequality_node', 'le_node', 'leq_node', 'imply_node', 'iff_node', 'not_node', 'or_node', 'and_node', 'xor_node', 'index_node', 'abs_node', 'division_node', 'max_node', 'min_node', 'modulo_node', 'pow_node', 'in_node', 'card_node', 'diff_node', 'intersect_node', 'subset_node', 'symdiff_node', 'union_node', 'acos_node', 'acosh_node', 'asin_node', 'asinh_node', 'atan_node', 'atanh_node', 'cos_node', 'cosh_node', 'sin_node', 'sinh_node', 'tan_node', 'tanh_node', 'div_node', 'exp_node', 'ln_node', 'log10_node', 'log2_node', 'sqrt_node', 'maximise_node', 'minimise_node', 'array_node', 'cumulatives_node', 'int_element_node', 'int_lin_eq_imp_node', 'array_int_maximum_node', 'schedule_unary_node', 'int_le_imp_node', 'global_cardinality_node', 'global_cardinality_low_up_node', 'maximum_arg_int_offset_node', 'circuit_node', 'count_eq_reif_node', 'set_in_imp_node', 'count_eq_node', 'global_cardinality_low_up_closed_node', 'bool_xor_imp_node', 'nooverlap_node', 'regular_node', 'all_different_node', 'eq_imp_node', 'all_equal_node', 'bool_element_node', 'array_int_minimum_node', 'bool_clause_reif_node', 'int_element2d_node', 'bin_packing_load_node', 'table_int_node', 'precede_node', 'array_int_lq_node', 'int_lin_le_imp_node', 'int_lin_ne_imp_node', 'increasing_int_node', 'inverse_offsets_node', 'nvalue_node', 'int_ne_imp_node', 'increasing_bool_node', 'member_int_node', 'table_int_imp_node', 'at_least_node', 'at_most_node', 'int_pow_node', 'global_cardinality_closed_node']

class Edge:
    def __init__(self, label:str) -> None:
        self.label = label

class Node:
    def __init__(self, label:str, _type:NODE_TYPE, value:Any=None) -> None:
        self.label = label
        self._type = _type
        self.value = value

    def __hash__(self) -> int:
        return hash(self.label + self._type)

    def __str__(self) -> str:
        return self.label

class Graph:
    def __init__(self) -> None:
        self.nodes:list[Node] = []
        self.edges:dict[tuple[int,int], Edge] = {}
        self.node_set:set = set()
        self._edge_to:dict[int,list[tuple[Node,Edge]]] = {}
        self._edge_from:dict[int,list[tuple[Node,Edge]]] = {}
        self.nodes_dict = {}

    def add_node(self, node:Node):
        h = hash(node)
        if h in self.node_set:
            return
        self.nodes_dict[h] = node
        self.node_set.add(h)
        self.nodes.append(node)

    def add_edge(self, _from:Node, _to:Node, e:Edge):
        assert hash(_from) in self.node_set, f'{_from} not in nodes'
        assert hash(_to) in self.node_set, f'{_to} not in nodes'
        self.edges[hash(_from),hash(_to)] = e
        if not _to in self._edge_to:
            self._edge_to[hash(_to)] = []
        if not _from in self._edge_from:
            self._edge_from[hash(_from)] = []
        self._edge_to[hash(_to)].append((_from, e))
        self._edge_from[hash(_from)].append((_to, e))

    def edge_to(self, node:Node) -> list[tuple[Node,Edge]]:
        return self._edge_to[hash(node)]

    def edge_from(self, node:Node) -> list[tuple[Node,Edge]]:
        return self._edge_from[hash(node)]

    @property
    def adjacency_matrix(self) -> list[list[int]]:
        n_nodes = len(self.nodes)
        matrix = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
        nidx = {hash(n):i for i,n in enumerate(self.nodes)}
        for (n1, n2) in self.edges.keys():
            matrix[nidx[hash(n1)]][nidx[hash(n2)]] = 1
        return matrix

    @property
    def edge_iterator(self) -> Iterable[tuple[tuple[Node,Node],Edge]]:
        for (n1, n2), v in self.edges.items():
            yield (self.nodes_dict[n1], self.nodes_dict[n2]), v

    def __str__(self) -> str:
        return str([str(n) for n in self.nodes]) + "\n" + "\n".join([" ".join([str(sm) for sm in m]) for m in self.adjacency_matrix])

    def to_nx(self) -> nx.Graph:
        g = nx.Graph()
        idxs = {}
        for idx, node in enumerate(self.nodes):
            idxs[hash(node)] = idx
            g.add_node(idx)
        for (n1, n2), e in self.edge_iterator:
            g.add_edge(idxs[hash(n1)],idxs[hash(n2)], attr=e.label)
        return g

def load_graph(file:str|TextIO) -> Graph:
    fp = file
    if isinstance(file, str):
        fp = open(file)

    assert fp.readable(), f'file {fp} not readable'

    nodes_lines = False
    edges_lines = False
    graph = Graph()
    nodes = {}
    for line in fp.readlines():
        line = line.strip()
        if line == 'nodes:':
            nodes_lines = True
            assert not edges_lines
            continue
        if line == 'edges:':
            assert nodes_lines
            nodes_lines = False
            edges_lines = True
            continue

        if nodes_lines:
            """The structure is: 'idx: label -- type -- extra'
               Extra is dependent on the type:
                - literal_node have: value -- type
                - var_node have: domain -- type
                - par_node have: value -- type
                - other nodes do not have extra
            """
            [idx, components_str] = line.split(': ')
            components = components_str.split(' -- ')
            label = components[0]
            node_type = components[1]
            if node_type == 'literal_node':
                node = Node(label=label, _type=node_type, value=(components[2], components[3]))
            elif node_type == 'var_node':
                node = Node(label=label, _type=node_type, value=(components[2], components[3]))
            elif node_type == 'par_node':
                node = Node(label=label, _type=node_type, value=(components[2], components[3]))
            else:
                node = Node(label=label, _type=node_type)
            graph.add_node(node)
            nodes[idx] = node
        if edges_lines:
            """The structure is: 'idx: idx_node1--idx_node2--edge_label'"""
            [idx, components_str] = line.split(': ')
            [idx_node1, idx_node2, label] = components_str.split('--')
            graph.add_edge(_from=nodes[idx_node1], _to=nodes[idx_node2], e=Edge(label))

    return graph
