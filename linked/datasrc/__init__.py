"""
Utils to generate graph data from properties and mini-languages.
"""

from linked.datasrc.minidot import mini_dot_to_graph_jdict
from linked.datasrc.vectors import (
    knn_graph,
    mutual_knn_graph,
    adaptive_knn_graph,
    epsilon_graph,
    random_graph,
)
