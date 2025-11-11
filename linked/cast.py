"""
Graph representation conversion system using castgraph.

This module provides a centralized TransformationGraph for converting between
various graph data representations. Each datasrc module registers its transformations
here, creating a unified conversion service.

Core graph kinds
----------------
- 'nodes_and_links': Dict with 'nodes' and 'links' keys (standard JSON graph format)
- 'edgelist': numpy array of shape (n_edges, 2) with [source, target] indices
- 'weighted_edgelist': numpy array of shape (n_edges, 3) with [source, target, weight]
- 'minidot': String in mini-dot format (e.g., "1 -> 2\\n2, 3 -> 5")
- 'adjacency_matrix': Dense numpy array of shape (n_nodes, n_nodes)
- 'sparse_adjacency': scipy sparse matrix
- 'adjacency_list': Dict mapping node indices to lists of neighbors
- 'networkx_graph': NetworkX Graph object
- 'networkx_digraph': NetworkX DiGraph object
- 'edges_dataframe': pandas DataFrame with source/target columns
- 'nodes_dataframe': pandas DataFrame with node metadata
- 'graph_dataframes': Dict with 'edges' and 'nodes' DataFrames

Design
------
Each datasrc module should:
1. Define kind predicates (isa functions) for its graph representations
2. Register transformations to/from at least one "hub" format (typically 'nodes_and_links' or 'edgelist')
3. Use contextlib.suppress to gracefully handle missing dependencies

Example usage
-------------
>>> from linked.cast import graph_transformer
>>> # Convert mini-dot string to NetworkX graph
>>> minidot = "1 -> 2\\n2, 3 -> 5, 6"
>>> nx_graph = graph_transformer.transform_any(minidot, 'networkx_graph')

>>> # Get a transformer function for repeated conversions
>>> to_edgelist = graph_transformer.get_transformer('minidot', 'edgelist')
>>> edgelist = to_edgelist(minidot)
"""

from contextlib import suppress
from typing import Any
import numpy as np

# Import the TransformationGraph from i2.castgraph
with suppress(ImportError, ModuleNotFoundError):
    from i2.castgraph import TransformationGraph

# Create the global graph transformation registry
graph_transformer = TransformationGraph()


# ============================================================================
# Core kind predicates (basic graph representations)
# ============================================================================


def _is_nodes_and_links_dict(obj: Any) -> bool:
    """Check if object is a nodes_and_links dict format."""
    if not isinstance(obj, dict):
        return False
    return 'nodes' in obj and 'links' in obj


def _is_edgelist(obj: Any) -> bool:
    """Check if object is an edge list (2-column numpy array)."""
    if not isinstance(obj, np.ndarray):
        return False
    return obj.ndim == 2 and obj.shape[1] == 2


def _is_weighted_edgelist(obj: Any) -> bool:
    """Check if object is a weighted edge list (3-column numpy array)."""
    if not isinstance(obj, np.ndarray):
        return False
    return obj.ndim == 2 and obj.shape[1] == 3


def _is_minidot_string(obj: Any) -> bool:
    """Check if object is a mini-dot format string."""
    if not isinstance(obj, str):
        return False
    # Simple heuristic: contains '->' which is the mini-dot edge operator
    return '->' in obj


def _is_adjacency_matrix(obj: Any) -> bool:
    """Check if object is a dense adjacency matrix."""
    if not isinstance(obj, np.ndarray):
        return False
    return obj.ndim == 2 and obj.shape[0] == obj.shape[1]


# Register core graph kinds
graph_transformer.add_node('nodes_and_links', isa=_is_nodes_and_links_dict)
graph_transformer.add_node('edgelist', isa=_is_edgelist)
graph_transformer.add_node('weighted_edgelist', isa=_is_weighted_edgelist)
graph_transformer.add_node('minidot', isa=_is_minidot_string)
graph_transformer.add_node('adjacency_matrix', isa=_is_adjacency_matrix)


# ============================================================================
# Utility functions for converters
# ============================================================================


def _ensure_node_id_field(nodes: list, id_field: str = 'id') -> list:
    """Ensure all nodes have an id field."""
    result = []
    for node in nodes:
        if isinstance(node, dict):
            if id_field not in node:
                # Try to infer id from other fields or use index
                node = {id_field: str(node), **node}
            result.append(node)
        else:
            # Convert bare values to dict with id
            result.append({id_field: str(node)})
    return result


def _normalize_nodes_and_links(
    obj: dict,
    *,
    id_field: str = 'id',
    source_field: str = 'source',
    target_field: str = 'target'
) -> dict:
    """Normalize a nodes_and_links dict to standard field names."""
    nodes = obj.get('nodes', [])
    links = obj.get('links', [])

    # Ensure nodes have id field
    nodes = _ensure_node_id_field(nodes, id_field)

    return {'nodes': nodes, 'links': links}


# ============================================================================
# Conversion hooks for datasrc modules to register themselves
# ============================================================================


def register_kind(kind: str, isa: callable = None):
    """
    Register a new graph kind with optional predicate.

    Parameters
    ----------
    kind : str
        The kind identifier
    isa : callable, optional
        Predicate function to detect if an object is of this kind

    Examples
    --------
    >>> with suppress(Exception):
    ...     register_kind('my_format', lambda x: isinstance(x, MyFormat))
    """
    graph_transformer.add_node(kind, isa=isa)


def register_transformation(src_kind: str, dst_kind: str, cost: float = 1.0):
    """
    Decorator to register a transformation between graph kinds.

    Parameters
    ----------
    src_kind : str
        Source graph kind
    dst_kind : str
        Destination graph kind
    cost : float
        Cost of this transformation (lower = preferred)

    Examples
    --------
    >>> @register_transformation('my_format', 'edgelist')
    ... def my_format_to_edgelist(obj, ctx):
    ...     return convert_to_edgelist(obj)
    """
    return graph_transformer.register_edge(src_kind, dst_kind, cost=cost)


# ============================================================================
# Module loading: Import datasrc modules to trigger their registrations
# ============================================================================


def _load_datasrc_modules():
    """
    Import all datasrc modules to trigger their converter registrations.

    Uses suppress to gracefully handle missing dependencies.
    """
    # Import order matters - hub formats first
    with suppress(ImportError, ModuleNotFoundError):
        from linked.datasrc import edgelist  # Core edge list conversions

    with suppress(ImportError, ModuleNotFoundError):
        from linked.datasrc import minidot  # Mini-dot conversions (already exists)

    with suppress(ImportError, ModuleNotFoundError):
        from linked.datasrc import vectors  # Vector-based graph construction

    with suppress(ImportError, ModuleNotFoundError):
        from linked.datasrc import networkx_graphs  # NetworkX conversions

    with suppress(ImportError, ModuleNotFoundError):
        from linked.datasrc import dataframes  # Pandas DataFrame conversions

    with suppress(ImportError, ModuleNotFoundError):
        from linked.datasrc import adjacency  # Adjacency matrix/list conversions


# Load all datasrc modules on import
_load_datasrc_modules()


# ============================================================================
# Convenience functions
# ============================================================================


def convert_graph(
    obj: Any, to_kind: str, from_kind: str = None, context: dict = None
) -> Any:
    """
    Convert a graph from one representation to another.

    Parameters
    ----------
    obj : Any
        The graph object to convert
    to_kind : str
        The target graph kind
    from_kind : str, optional
        The source graph kind (auto-detected if None)
    context : dict, optional
        Additional context for the conversion (e.g., column names, field mappings)

    Returns
    -------
    Any
        The converted graph object

    Examples
    --------
    >>> minidot = "1 -> 2\\n2 -> 3"
    >>> graph = convert_graph(minidot, 'nodes_and_links')
    >>> 'nodes' in graph and 'links' in graph
    True
    """
    if from_kind is None:
        return graph_transformer.transform_any(obj, to_kind, context=context)
    else:
        return graph_transformer.transform(
            obj, to_kind, from_kind=from_kind, context=context
        )


def graph_converter(from_kind: str, to_kind: str, context: dict = None) -> callable:
    """
    Get a converter function for repeated conversions.

    Parameters
    ----------
    from_kind : str
        Source graph kind
    to_kind : str
        Destination graph kind
    context : dict, optional
        Context to bake into the converter

    Returns
    -------
    callable
        A function that converts graphs from from_kind to to_kind

    Examples
    --------
    >>> to_edgelist = graph_converter('minidot', 'edgelist')
    >>> edgelist = to_edgelist("1 -> 2")
    """
    return graph_transformer.get_transformer(from_kind, to_kind, context=context)


def graph_kinds() -> set:
    """
    List all registered graph kinds.

    Returns
    -------
    set
        Set of all registered graph kind identifiers

    Examples
    --------
    >>> kinds = graph_kinds()
    >>> 'nodes_and_links' in kinds
    True
    """
    return graph_transformer.kinds()


def reachable_from_kind(kind: str) -> set:
    """
    Get all graph kinds reachable from a given kind.

    Parameters
    ----------
    kind : str
        The source graph kind

    Returns
    -------
    set
        Set of reachable graph kind identifiers

    Examples
    --------
    >>> reachable = reachable_from_kind('minidot')
    >>> 'nodes_and_links' in reachable
    True
    """
    return graph_transformer.reachable_from(kind)


def sources_for_kind(kind: str) -> set:
    """
    Get all graph kinds that can be converted to a given kind.

    Parameters
    ----------
    kind : str
        The target graph kind

    Returns
    -------
    set
        Set of source graph kind identifiers

    Examples
    --------
    >>> sources = sources_for_kind('edgelist')
    >>> len(sources) >= 0
    True
    """
    return graph_transformer.sources_for(kind)


# ============================================================================
# Attach utility functions to convert_graph for convenient access
# ============================================================================

convert_graph.graph_kinds = graph_kinds
convert_graph.graph_converter = graph_converter
convert_graph.reachable_from_kind = reachable_from_kind
convert_graph.sources_for_kind = sources_for_kind


# ============================================================================
# Export public API
# ============================================================================

__all__ = [
    'graph_transformer',
    'convert_graph',
    'graph_converter',
    'graph_kinds',
    'reachable_from_kind',
    'sources_for_kind',
    'register_kind',
    'register_transformation',
]
