import json
import os
import socket
import subprocess
import tempfile
from typing import Literal

try:
    from .graph_loader import Graph
except ImportError:
    from graph_loader import is_global, Graph, load_graph

import shutil

# Path to the compiled ZincToWl Julia executable binary
ZINC_TO_WL_BIN = shutil.which("ZincToWl") or os.path.expanduser("~/.local/bin/ZincToWl")

def get_effective_domain_size(domain: str, type_str: str) -> int:
    if type_str == 'bool' or domain in ['bool', 'false..true', 'true..false']:
        return 2
    if 'set' in type_str:
        if '..' in domain:
            parts = domain.split('..')
            try:
                return abs(int(parts[1]) - int(parts[0])) + 1
            except ValueError:
                pass
        if domain.startswith('{') and domain.endswith('}'):
            content = domain[1:-1].strip()
            if not content:
                return 0
            return len(content.split(','))
        try:
            return int(domain)
        except ValueError:
            return 1
    if '..' in domain:
        parts = domain.split('..')
        try:
            return abs(int(parts[1]) - int(parts[0])) + 1
        except ValueError:
            pass
    try:
        int(domain)
        return 1
    except ValueError:
        pass
    return 1

def _get_graph_filepath(graph: Graph | str) -> tuple[str, bool]:
    """Returns the file path for a graph object or string path, and a flag if a temp file was created."""
    if isinstance(graph, str) and os.path.exists(graph):
        return graph, False
    if hasattr(graph, 'filepath') and graph.filepath and os.path.exists(graph.filepath):
        return graph.filepath, False
    
    tfile = tempfile.NamedTemporaryFile(suffix='.graph', mode='w', delete=False)
    nodes_dict = {}
    tfile.write("nodes:\n")
    for idx, node in enumerate(graph.nodes):
        nodes_dict[hash(node)] = idx
        extra = ""
        if node.value is not None:
            if isinstance(node.value, tuple):
                extra = f" -- {' -- '.join(str(v) for v in node.value)}"
            else:
                extra = f" -- {node.value}"
        tfile.write(f"{idx}: {node.label} -- {node._type}{extra}\n")
    
    tfile.write("edges:\n")
    for e_idx, ((n1, n2), edge) in enumerate(graph.edge_iterator):
        idx1 = nodes_dict[hash(n1)]
        idx2 = nodes_dict[hash(n2)]
        tfile.write(f"{e_idx}: {idx1}--{idx2}--{edge.label}\n")
    
    tfile.close()
    return tfile.name, True

def _resolve_colors_path(colors: str | dict | None) -> str:
    """Resolves the colors file path for the external ZincToWl binary."""
    if isinstance(colors, str) and colors:
        path = colors
    elif isinstance(colors, dict) and '_bin_path' in colors:
        path = colors['_bin_path']
    elif isinstance(colors, dict) and 'path' in colors:
        path = colors['path']
    else:
        path = 'colors.bin'

    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) == 0:
        try:
            os.remove(path)
        except OSError:
            pass
    return path

def send_to_server(socket_path: str, args: list[str]) -> str:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(socket_path)
    msg = "\0".join(args) + "\0\n"
    s.sendall(msg.encode())
    all_data = ""
    while True:
        data = s.recv(4096)
        if not data:
            break
        all_data += data.decode()
    s.close()
    return all_data[all_data.index('{'):]

def _run_zinc_to_wl(
    graph: Graph | str,
    colors: str | dict | None,
    method: str,
    max_iter: int = 1,
    training: bool = True,
    socket_path: str = "/tmp/zinctowl_8.sock"
) -> tuple[dict[str, int], dict]:
    graph_path, created_temp = _get_graph_filepath(graph)
    target_graph_path = graph_path

    colors_bin = _resolve_colors_path(colors)

    cmd = [
        target_graph_path,
        '-m', method,
        '-k', str(max_iter),
        '--colors', colors_bin,
        '-t', 'true' if training else 'false',
        '-c', '8'
    ]

    data = ""
    if os.path.exists(socket_path):
        try:
            data = send_to_server(socket_path, cmd)
        except Exception:
            pass

    if not data:
        sub_cmd = [ZINC_TO_WL_BIN] + cmd
        proc = subprocess.run(sub_cmd, capture_output=True, text=True, check=True)
        data = proc.stdout

    if created_temp and os.path.exists(graph_path):
        try:
            os.remove(graph_path)
        except OSError:
            pass

    raw_text = data.strip()
    if not raw_text:
        raise ValueError(f"ZincToWl produced no output for method {method} on {graph_path}")
    
    try:
        res_json = json.loads(raw_text)
    except Exception as e:
        print(target_graph_path)
        print(raw_text)
        raise e

    scalar_fields = {
        'n_nodes', 'cpv', 'cpp', 'd_ratio_int_vars', 'd_ratio_bool_vars',
        'o_deg_cons', 'o_deg_std', 'o_dom_deg', 'v_ent_deg_vars', 'v_sum_domdeg_vars'
    }

    color_counts: dict[str, int] = {}
    globals_pairs: dict[tuple[str, str], int] = {}
    extra_info = {
        'n_nodes': int(res_json.get('n_nodes', 0)),
        'cpv': float(res_json.get('cpv', 0.0)),
        'cpp': float(res_json.get('cpp', 0.0)),
        'd_ratio_int_vars': float(res_json.get('d_ratio_int_vars', 0.0)),
        'd_ratio_bool_vars': float(res_json.get('d_ratio_bool_vars', 0.0)),
        'o_deg_cons': float(res_json.get('o_deg_cons', 0.0)),
        'o_deg_std': float(res_json.get('o_deg_std', 0.0)),
        'o_dom_deg': float(res_json.get('o_dom_deg', 0.0)),
        'v_ent_deg_vars': float(res_json.get('v_ent_deg_vars', 0.0)),
        'v_sum_domdeg_vars': float(res_json.get('v_sum_domdeg_vars', 0.0)),
        'globals_pairs': globals_pairs
    }

    for k, v in res_json.items():
        if k in scalar_fields:
            continue
        elif k.startswith('(') and k.endswith(')'):
            inner = k[1:-1]
            if ',' in inner:
                t1, t2 = inner.split(',', 1)
                globals_pairs[(t1.strip(), t2.strip())] = int(v)
            else:
                globals_pairs[(k, "")] = int(v)
        else:
            color_counts[str(k)] = int(v)

    return color_counts, extra_info

def standard_wl(graph: Graph | str, colors: str | dict | None = 'colors.bin', max_iter: int = 10, training: bool = True) -> dict[str, int]:
    color_counts, _ = _run_zinc_to_wl(graph, colors, 'wl', max_iter=max_iter, training=training)
    return color_counts

def wl_with_node_features(graph: Graph | str, colors: str | dict | None = 'colors.bin', max_iter: int = 10, training: bool = True) -> dict[str, int]:
    color_counts, _ = _run_zinc_to_wl(graph, colors, 'wl-n', max_iter=max_iter, training=training)
    return color_counts

def undirected_wl_with_node_features(graph: Graph | str, colors: str | dict | None = 'colors.bin', max_iter: int = 10, training: bool = True) -> dict[str, int]:
    color_counts, _ = _run_zinc_to_wl(graph, colors, 'wl-un', max_iter=max_iter, training=training)
    return color_counts

def wl_with_edge_features(graph: Graph | str, colors: str | dict | None = 'colors.bin', max_iter: int = 10, training: bool = True) -> dict[str, int]:
    color_counts, _ = _run_zinc_to_wl(graph, colors, 'wl-e', max_iter=max_iter, training=training)
    return color_counts

def wl_with_node_and_edge_features(graph: Graph | str, colors: str | dict | None = 'colors.bin', max_iter: int = 10, training: bool = True) -> dict[str, int]:
    color_counts, _ = _run_zinc_to_wl(graph, colors, 'wl-ne', max_iter=max_iter, training=training)
    return color_counts

def undirected_wl_with_node_and_edge_features(graph: Graph | str, colors: str | dict | None = 'colors.bin', max_iter: int = 10, training: bool = True) -> dict[str, int]:
    color_counts, _ = _run_zinc_to_wl(graph, colors, 'wl-une', max_iter=max_iter, training=training)
    return color_counts

def wl_extended_features(graph: Graph | str, colors: str | dict | None = 'colors.bin', max_iter: int = 1, training: bool = True) -> tuple[dict[str, int], dict]:
    return _run_zinc_to_wl(graph, colors, 'wl-nc', max_iter=max_iter, training=training)

def undirected_wl_extended_features(graph: Graph | str, colors: str | dict | None = 'colors.bin', max_iter: int = 1, training: bool = True) -> tuple[dict[str, int], dict]:
    return _run_zinc_to_wl(graph, colors, 'wl-unc', max_iter=max_iter, training=training)

def wl_extended_features_with_edges(graph: Graph | str, colors: str | dict | None = 'colors.bin', max_iter: int = 1, training: bool = True) -> tuple[dict[str, int], dict]:
    return _run_zinc_to_wl(graph, colors, 'wl-nec', max_iter=max_iter, training=training)

def undirected_wl_extended_features_with_edges(graph: Graph | str, colors: str | dict | None = 'colors.bin', max_iter: int = 1, training: bool = True) -> tuple[dict[str, int], dict]:
    return _run_zinc_to_wl(graph, colors, 'wl-unec', max_iter=max_iter, training=training)

def wl_features(graph: Graph | str,
                colors: str | dict | None = 'colors.bin',
                max_iter: int = 10,
                training: bool = True,
                wl_type: Literal['standard', 'node_features', 'edge_features', 'node_edge_features'] = 'standard',
                max_colors: int | None = None) -> dict[str, int]:
    if wl_type == 'standard':
        return standard_wl(graph, colors, max_iter, training)
    elif wl_type == 'edge_features':
        return wl_with_edge_features(graph, colors, max_iter, training)
    elif wl_type == 'node_features':
        return wl_with_node_features(graph, colors, max_iter, training)
    elif wl_type == 'node_edge_features':
        return wl_with_node_and_edge_features(graph, colors, max_iter, training)

    raise Exception(f'unrecognised wl_type: {wl_type}')

if __name__ == '__main__':
    graph_path = './data/graphs/accap-2019-accap_instance3.graph'
    if os.path.exists(graph_path):
        res, extra = wl_extended_features(graph_path, '/tmp/test_colors.bin')
        print("WL color counts sample:", list(res.items())[:5])
        print("Extended features extra_info:", extra)

