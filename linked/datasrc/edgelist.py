"""
Edge list graph representations and conversions.

An edge list is one of the most compact graph representations, consisting of
an array of edges where each edge is [source, target] or [source, target, weight].

Kinds defined here
------------------
- 'edgelist': numpy array (n_edges, 2) with integer node indices
- 'weighted_edgelist': numpy array (n_edges, 3) with [source, target, weight]

Conversions
-----------
- edgelist <-> nodes_and_links
- weighted_edgelist <-> nodes_and_links
- edgelist <-> weighted_edgelist (add/remove weights)
"""

from contextlib import suppress
import numpy as np

# Only register if cast module is available
with suppress(ImportError, ModuleNotFoundError):
    from linked.cast import register_transformation


# ============================================================================
# Edge list <-> nodes_and_links conversions
# ============================================================================


def edgelist_to_nodes_and_links(edgelist: np.ndarray, ctx: dict = None) -> dict:
    """
    Convert edge list to nodes_and_links dict.

    Parameters
    ----------
    edgelist : np.ndarray
        Array of shape (n_edges, 2) or (n_edges, 3)
    ctx : dict, optional
        Context with optional keys:
        - 'id_field': field name for node id (default: 'id')
        - 'source_field': field name for edge source (default: 'source')
        - 'target_field': field name for edge target (default: 'target')
        - 'weight_field': field name for edge weight (default: 'weight')
        - 'node_id_map': dict mapping indices to node ids

    Returns
    -------
    dict
        Dict with 'nodes' and 'links' keys

    Examples
    --------
    >>> edgelist = np.array([[0, 1], [1, 2], [2, 0]])
    >>> result = edgelist_to_nodes_and_links(edgelist)
    >>> len(result['nodes'])
    3
    >>> len(result['links'])
    3
    """
    ctx = ctx or {}
    id_field = ctx.get('id_field', 'id')
    source_field = ctx.get('source_field', 'source')
    target_field = ctx.get('target_field', 'target')
    weight_field = ctx.get('weight_field', 'weight')
    node_id_map = ctx.get('node_id_map', None)

    if len(edgelist) == 0:
        return {'nodes': [], 'links': []}

    # Extract unique node indices
    node_indices = np.unique(edgelist[:, :2].astype(int))

    # Create nodes
    if node_id_map:
        nodes = [{id_field: node_id_map.get(idx, str(idx))} for idx in node_indices]
    else:
        nodes = [{id_field: str(idx)} for idx in node_indices]

    # Create links
    has_weights = edgelist.shape[1] >= 3
    links = []

    for edge in edgelist:
        src_idx = int(edge[0])
        tgt_idx = int(edge[1])

        link = {
            source_field: (
                node_id_map.get(src_idx, str(src_idx)) if node_id_map else str(src_idx)
            ),
            target_field: (
                node_id_map.get(tgt_idx, str(tgt_idx)) if node_id_map else str(tgt_idx)
            ),
        }

        if has_weights:
            link[weight_field] = float(edge[2])

        links.append(link)

    return {'nodes': nodes, 'links': links}


def nodes_and_links_to_edgelist(nodes_and_links: dict, ctx: dict = None) -> np.ndarray:
    """
    Convert nodes_and_links dict to edge list.

    Parameters
    ----------
    nodes_and_links : dict
        Dict with 'nodes' and 'links' keys
    ctx : dict, optional
        Context with optional keys:
        - 'id_field': field name for node id (default: 'id')
        - 'source_field': field name for edge source (default: 'source')
        - 'target_field': field name for edge target (default: 'target')
        - 'weight_field': field name for edge weight (default: 'weight')
        - 'include_weights': whether to include weights if present (default: False)

    Returns
    -------
    np.ndarray
        Edge list array of shape (n_edges, 2) or (n_edges, 3)

    Examples
    --------
    >>> graph = {'nodes': [{'id': '0'}, {'id': '1'}], 'links': [{'source': '0', 'target': '1'}]}
    >>> edgelist = nodes_and_links_to_edgelist(graph)
    >>> edgelist.shape
    (1, 2)
    """
    ctx = ctx or {}
    id_field = ctx.get('id_field', 'id')
    source_field = ctx.get('source_field', 'source')
    target_field = ctx.get('target_field', 'target')
    weight_field = ctx.get('weight_field', 'weight')
    include_weights = ctx.get('include_weights', False)

    nodes = nodes_and_links.get('nodes', [])
    links = nodes_and_links.get('links', [])

    if not links:
        return np.empty((0, 3 if include_weights else 2))

    # Build node id to index mapping
    node_id_to_idx = {}
    for idx, node in enumerate(nodes):
        if isinstance(node, dict):
            node_id = node.get(id_field, str(idx))
        else:
            node_id = str(node)
        node_id_to_idx[str(node_id)] = idx

    # Build edge list
    edges = []
    has_any_weights = any(
        weight_field in link for link in links if isinstance(link, dict)
    )
    should_include_weights = include_weights and has_any_weights

    for link in links:
        if not isinstance(link, dict):
            continue

        src_id = str(link.get(source_field, ''))
        tgt_id = str(link.get(target_field, ''))

        # Get or create indices for source and target
        if src_id not in node_id_to_idx:
            node_id_to_idx[src_id] = len(node_id_to_idx)
        if tgt_id not in node_id_to_idx:
            node_id_to_idx[tgt_id] = len(node_id_to_idx)

        src_idx = node_id_to_idx[src_id]
        tgt_idx = node_id_to_idx[tgt_id]

        if should_include_weights:
            weight = link.get(weight_field, 1.0)
            edges.append([src_idx, tgt_idx, weight])
        else:
            edges.append([src_idx, tgt_idx])

    return (
        np.array(edges) if edges else np.empty((0, 3 if should_include_weights else 2))
    )


# ============================================================================
# Weighted <-> unweighted edge list conversions
# ============================================================================


def edgelist_to_weighted_edgelist(edgelist: np.ndarray, ctx: dict = None) -> np.ndarray:
    """
    Add default weights to an unweighted edge list.

    Parameters
    ----------
    edgelist : np.ndarray
        Array of shape (n_edges, 2)
    ctx : dict, optional
        Context with 'default_weight' key (default: 1.0)

    Returns
    -------
    np.ndarray
        Array of shape (n_edges, 3)

    Examples
    --------
    >>> edgelist = np.array([[0, 1], [1, 2]])
    >>> weighted = edgelist_to_weighted_edgelist(edgelist)
    >>> weighted.shape
    (2, 3)
    >>> float(weighted[0, 2])
    1.0
    """
    ctx = ctx or {}
    default_weight = ctx.get('default_weight', 1.0)

    if len(edgelist) == 0:
        return np.empty((0, 3))

    weights = np.full((len(edgelist), 1), default_weight)
    return np.column_stack([edgelist, weights])


def weighted_edgelist_to_edgelist(
    weighted_edgelist: np.ndarray, ctx: dict = None
) -> np.ndarray:
    """
    Remove weights from a weighted edge list.

    Parameters
    ----------
    weighted_edgelist : np.ndarray
        Array of shape (n_edges, 3)
    ctx : dict, optional
        Not used

    Returns
    -------
    np.ndarray
        Array of shape (n_edges, 2)

    Examples
    --------
    >>> weighted = np.array([[0, 1, 0.5], [1, 2, 0.8]])
    >>> edgelist = weighted_edgelist_to_edgelist(weighted)
    >>> edgelist.shape
    (2, 2)
    """
    if len(weighted_edgelist) == 0:
        return np.empty((0, 2), dtype=int)

    return weighted_edgelist[:, :2].astype(int)


# ============================================================================
# Register transformations with cast module
# ============================================================================

with suppress(ImportError, ModuleNotFoundError):
    # Edge list <-> nodes_and_links
    register_transformation('edgelist', 'nodes_and_links', cost=0.5)(
        edgelist_to_nodes_and_links
    )
    register_transformation('nodes_and_links', 'edgelist', cost=0.5)(
        nodes_and_links_to_edgelist
    )

    # Weighted edge list <-> nodes_and_links (same functions work for both)
    register_transformation('weighted_edgelist', 'nodes_and_links', cost=0.5)(
        edgelist_to_nodes_and_links
    )

    # For nodes_and_links -> weighted_edgelist, we need to ensure weights are included
    @register_transformation('nodes_and_links', 'weighted_edgelist', cost=0.5)
    def _nodes_and_links_to_weighted_edgelist(obj, ctx):
        ctx = ctx or {}
        ctx['include_weights'] = True
        return nodes_and_links_to_edgelist(obj, ctx)

    # Weighted <-> unweighted conversions
    register_transformation('edgelist', 'weighted_edgelist', cost=0.1)(
        edgelist_to_weighted_edgelist
    )
    register_transformation('weighted_edgelist', 'edgelist', cost=0.1)(
        weighted_edgelist_to_edgelist
    )
