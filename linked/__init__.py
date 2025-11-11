"""
Create and transform graph data.

The linked package provides tools for creating and converting between various
graph data representations. It includes:

- Graph construction from vectors (k-NN graphs, mutual k-NN, etc.)
- Mini-dot language for simple graph specification
- Comprehensive graph format conversion using castgraph
- Support for NetworkX, pandas DataFrames, adjacency matrices, edge lists, and more

Examples:

>>> from linked import mini_dot_to_graph_jdict
>>> mini_dot_to_graph_jdict('''
... 1 -> 2
... 2, 3 -> 5, 6, 7
... ''')  # doctest: +NORMALIZE_WHITESPACE
{'nodes': [{'id': '1'},
{'id': '2'},
{'id': '5'},
{'id': '6'},
{'id': '7'},
{'id': '3'}],
'links': [{'source': '1', 'target': '2'},
{'source': '2', 'target': '5'},
{'source': '2', 'target': '6'},
{'source': '2', 'target': '7'},
{'source': '3', 'target': '5'},
{'source': '3', 'target': '6'},
{'source': '3', 'target': '7'}]}

Graph conversion examples:

>>> from linked import convert_graph
>>> import numpy as np
>>> # Convert edge list to nodes_and_links format
>>> edgelist = np.array([[0, 1], [1, 2]])
>>> graph = convert_graph(edgelist, 'nodes_and_links')
>>> 'nodes' in graph and 'links' in graph
True

>>> # Convert mini-dot to edge list
>>> minidot = "1 -> 2\\n2 -> 3"
>>> edgelist = convert_graph(minidot, 'edgelist', from_kind='minidot')
>>> len(edgelist) == 2
True
"""

from linked.datasrc import (
    mini_dot_to_graph_jdict,
    knn_graph,
    mutual_knn_graph,
    adaptive_knn_graph,
    epsilon_graph,
    random_graph,
)

# Import cast API
from linked.cast import (
    convert_graph,
    graph_converter,
    graph_kinds,
    reachable_from_kind,
    sources_for_kind,
    graph_transformer,
)

__all__ = [
    # Legacy datasrc exports
    'mini_dot_to_graph_jdict',
    'knn_graph',
    'mutual_knn_graph',
    'adaptive_knn_graph',
    'epsilon_graph',
    'random_graph',
    # Graph conversion API
    'convert_graph',
    'graph_converter',
    'graph_kinds',
    'reachable_from_kind',
    'sources_for_kind',
    'graph_transformer',
]
