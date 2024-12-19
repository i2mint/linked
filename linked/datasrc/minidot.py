"""
mini-dot: A very simple mini-language resembling graphviz dot
"""

from operator import methodcaller
from functools import partial
from itertools import product, chain
from typing import Iterable
import re
from types import SimpleNamespace

from dol import Pipe

DFLT_FIELD_NAMES = {
    'id': 'id',
    'nodes': 'nodes',
    'links': 'links',
    'source': 'source',
    'target': 'target',
}

comment_marker = r'#'
to_and_from_nodes_sep = '->'
node_regular_expression = r'\w+'

iterize = lambda func: partial(map, func)

remove_comments = Pipe(
    re.compile(f'[^{comment_marker}]*').match, methodcaller('group', 0)
)
split_to_and_from_nodes = methodcaller('split', to_and_from_nodes_sep)
extract_nodes = re.compile(node_regular_expression).findall

process_one_line = Pipe(
    source_and_target_lists_str=split_to_and_from_nodes,
    source_and_target_lists=Pipe(iterize(extract_nodes), list),
    generate_all_combinations=lambda x: product(*x),
)

get_source_target_pairs = Pipe(
    lines=lambda x: str.splitlines(x),
    remove_comments=iterize(remove_comments),
    filter_out_empty_lines=partial(filter, None),
    process_lines=iterize(process_one_line),
    chain=chain.from_iterable,
)


def source_target_pairs_to_graph_jdict(
    source_target_pairs: Iterable, *, field_names=DFLT_FIELD_NAMES
):
    nodes = []
    links = []
    _nodes = set(nodes)

    def add_node_if_not_already_there(node):
        if node not in _nodes:
            nodes.append(node)
            _nodes.add(node)

    field = SimpleNamespace(**dict(DFLT_FIELD_NAMES, **field_names))

    for source, target in source_target_pairs:
        add_node_if_not_already_there(source)
        add_node_if_not_already_there(target)
        links.append({field.source: source, field.target: target})

    return {field.nodes: [{field.id: node} for node in nodes], field.links: links}


def mini_dot_to_graph_jdict(mini_dot: str, *, field_names=DFLT_FIELD_NAMES):
    """
    Make a graph json dict from a mini-dot string.

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
    """
    pairs = get_source_target_pairs(mini_dot)
    return source_target_pairs_to_graph_jdict(pairs, field_names=field_names)
