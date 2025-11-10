"""
NetworkX graph representations and conversions.

NetworkX is one of the most popular Python graph libraries, supporting both
undirected (Graph) and directed (DiGraph) graphs with extensive algorithms.

Kinds defined here
------------------
- 'networkx_graph': networkx.Graph (undirected)
- 'networkx_digraph': networkx.DiGraph (directed)

Conversions
-----------
- networkx graphs <-> edgelist
- networkx graphs <-> nodes_and_links
"""

from contextlib import suppress
import numpy as np

# Try to import networkx - only register if available
_has_networkx = False
with suppress(ImportError, ModuleNotFoundError):
    import networkx as nx

    _has_networkx = True

# Only register if both networkx and cast are available
with suppress(ImportError, ModuleNotFoundError):
    from linked.cast import register_kind, register_transformation


# ============================================================================
# Kind predicates
# ============================================================================

if _has_networkx:

    def _is_networkx_graph(obj) -> bool:
        """Check if object is a NetworkX Graph."""
        return isinstance(obj, nx.Graph) and not isinstance(obj, nx.DiGraph)

    def _is_networkx_digraph(obj) -> bool:
        """Check if object is a NetworkX DiGraph."""
        return isinstance(obj, nx.DiGraph)

    # Register kinds
    with suppress(ImportError, ModuleNotFoundError):
        register_kind('networkx_graph', isa=_is_networkx_graph)
        register_kind('networkx_digraph', isa=_is_networkx_digraph)


# ============================================================================
# NetworkX <-> edge list conversions
# ============================================================================

if _has_networkx:

    def networkx_to_edgelist(graph: nx.Graph, ctx: dict = None) -> np.ndarray:
        """
        Convert NetworkX graph to edge list.

        Parameters
        ----------
        graph : nx.Graph or nx.DiGraph
            NetworkX graph object
        ctx : dict, optional
            Context with optional keys:
            - 'weight_attr': attribute name for edge weight (default: 'weight')
            - 'include_weights': whether to include weights (default: auto-detect)

        Returns
        -------
        np.ndarray
            Edge list array

        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_edges_from([(0, 1), (1, 2)])
        >>> edgelist = networkx_to_edgelist(G)
        >>> edgelist.shape[0] == 2
        True
        """
        ctx = ctx or {}
        weight_attr = ctx.get('weight_attr', 'weight')
        include_weights = ctx.get('include_weights', None)

        if graph.number_of_edges() == 0:
            return np.empty((0, 2))

        # Auto-detect if weights should be included
        if include_weights is None:
            # Check if any edge has weight attribute
            include_weights = any(
                weight_attr in data for _, _, data in graph.edges(data=True)
            )

        # Build node to index mapping
        node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

        # Extract edges
        edges = []
        for src, tgt, data in graph.edges(data=True):
            src_idx = node_to_idx[src]
            tgt_idx = node_to_idx[tgt]

            if include_weights:
                weight = data.get(weight_attr, 1.0)
                edges.append([src_idx, tgt_idx, weight])
            else:
                edges.append([src_idx, tgt_idx])

        return np.array(edges) if edges else np.empty((0, 3 if include_weights else 2))

    def edgelist_to_networkx_graph(edgelist: np.ndarray, ctx: dict = None) -> nx.Graph:
        """
        Convert edge list to NetworkX undirected graph.

        Parameters
        ----------
        edgelist : np.ndarray
            Edge list array of shape (n_edges, 2) or (n_edges, 3)
        ctx : dict, optional
            Context with optional keys:
            - 'weight_attr': attribute name for edge weight (default: 'weight')
            - 'node_id_map': dict mapping indices to node ids

        Returns
        -------
        nx.Graph
            NetworkX undirected graph

        Examples
        --------
        >>> edgelist = np.array([[0, 1], [1, 2]])
        >>> G = edgelist_to_networkx_graph(edgelist)
        >>> G.number_of_nodes()
        3
        """
        ctx = ctx or {}
        weight_attr = ctx.get('weight_attr', 'weight')
        node_id_map = ctx.get('node_id_map', None)

        G = nx.Graph()

        if len(edgelist) == 0:
            return G

        # Add nodes
        node_indices = np.unique(edgelist[:, :2].astype(int))
        if node_id_map:
            G.add_nodes_from(node_id_map.get(idx, idx) for idx in node_indices)
        else:
            G.add_nodes_from(node_indices)

        # Add edges
        has_weights = edgelist.shape[1] >= 3
        for edge in edgelist:
            src = int(edge[0])
            tgt = int(edge[1])

            if node_id_map:
                src = node_id_map.get(src, src)
                tgt = node_id_map.get(tgt, tgt)

            if has_weights:
                G.add_edge(src, tgt, **{weight_attr: float(edge[2])})
            else:
                G.add_edge(src, tgt)

        return G

    def edgelist_to_networkx_digraph(
        edgelist: np.ndarray, ctx: dict = None
    ) -> nx.DiGraph:
        """
        Convert edge list to NetworkX directed graph.

        Parameters
        ----------
        edgelist : np.ndarray
            Edge list array
        ctx : dict, optional
            Context (same as edgelist_to_networkx_graph)

        Returns
        -------
        nx.DiGraph
            NetworkX directed graph

        Examples
        --------
        >>> edgelist = np.array([[0, 1], [1, 2]])
        >>> G = edgelist_to_networkx_digraph(edgelist)
        >>> G.is_directed()
        True
        """
        ctx = ctx or {}
        weight_attr = ctx.get('weight_attr', 'weight')
        node_id_map = ctx.get('node_id_map', None)

        G = nx.DiGraph()

        if len(edgelist) == 0:
            return G

        # Add nodes
        node_indices = np.unique(edgelist[:, :2].astype(int))
        if node_id_map:
            G.add_nodes_from(node_id_map.get(idx, idx) for idx in node_indices)
        else:
            G.add_nodes_from(node_indices)

        # Add edges
        has_weights = edgelist.shape[1] >= 3
        for edge in edgelist:
            src = int(edge[0])
            tgt = int(edge[1])

            if node_id_map:
                src = node_id_map.get(src, src)
                tgt = node_id_map.get(tgt, tgt)

            if has_weights:
                G.add_edge(src, tgt, **{weight_attr: float(edge[2])})
            else:
                G.add_edge(src, tgt)

        return G


# ============================================================================
# NetworkX <-> nodes_and_links conversions
# ============================================================================

if _has_networkx:

    def networkx_to_nodes_and_links(graph: nx.Graph, ctx: dict = None) -> dict:
        """
        Convert NetworkX graph to nodes_and_links dict.

        Parameters
        ----------
        graph : nx.Graph or nx.DiGraph
            NetworkX graph
        ctx : dict, optional
            Context with optional keys:
            - 'id_field': field name for node id (default: 'id')
            - 'source_field': field name for edge source (default: 'source')
            - 'target_field': field name for edge target (default: 'target')
            - 'weight_field': field name for edge weight (default: 'weight')
            - 'weight_attr': NetworkX attribute for weight (default: 'weight')
            - 'include_node_attrs': whether to include node attributes (default: True)
            - 'include_edge_attrs': whether to include edge attributes (default: True)

        Returns
        -------
        dict
            Dict with 'nodes' and 'links' keys

        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_edges_from([(0, 1), (1, 2)])
        >>> result = networkx_to_nodes_and_links(G)
        >>> len(result['nodes']) == 3
        True
        """
        ctx = ctx or {}
        id_field = ctx.get('id_field', 'id')
        source_field = ctx.get('source_field', 'source')
        target_field = ctx.get('target_field', 'target')
        weight_field = ctx.get('weight_field', 'weight')
        weight_attr = ctx.get('weight_attr', 'weight')
        include_node_attrs = ctx.get('include_node_attrs', True)
        include_edge_attrs = ctx.get('include_edge_attrs', True)

        # Build nodes
        nodes = []
        for node, data in graph.nodes(data=True):
            node_dict = {id_field: str(node)}
            if include_node_attrs and data:
                node_dict.update(data)
            nodes.append(node_dict)

        # Build links
        links = []
        for src, tgt, data in graph.edges(data=True):
            link = {
                source_field: str(src),
                target_field: str(tgt),
            }

            if include_edge_attrs and data:
                # Include weight specially if present
                if weight_attr in data:
                    link[weight_field] = data[weight_attr]
                # Include other attributes
                for key, value in data.items():
                    if key != weight_attr:
                        link[key] = value

            links.append(link)

        return {'nodes': nodes, 'links': links}

    def nodes_and_links_to_networkx_graph(
        nodes_and_links: dict, ctx: dict = None
    ) -> nx.Graph:
        """
        Convert nodes_and_links dict to NetworkX undirected graph.

        Parameters
        ----------
        nodes_and_links : dict
            Dict with 'nodes' and 'links' keys
        ctx : dict, optional
            Context (see networkx_to_nodes_and_links)

        Returns
        -------
        nx.Graph
            NetworkX undirected graph

        Examples
        --------
        >>> data = {'nodes': [{'id': '0'}, {'id': '1'}], 'links': [{'source': '0', 'target': '1'}]}
        >>> G = nodes_and_links_to_networkx_graph(data)
        >>> G.number_of_edges()
        1
        """
        ctx = ctx or {}
        id_field = ctx.get('id_field', 'id')
        source_field = ctx.get('source_field', 'source')
        target_field = ctx.get('target_field', 'target')

        G = nx.Graph()

        # Add nodes with attributes
        for node in nodes_and_links.get('nodes', []):
            if isinstance(node, dict):
                node_id = node.get(id_field)
                # Get other attributes (excluding id)
                attrs = {k: v for k, v in node.items() if k != id_field}
                G.add_node(node_id, **attrs)
            else:
                G.add_node(str(node))

        # Add edges with attributes
        for link in nodes_and_links.get('links', []):
            if isinstance(link, dict):
                src = link.get(source_field)
                tgt = link.get(target_field)
                # Get edge attributes (excluding source and target)
                attrs = {
                    k: v
                    for k, v in link.items()
                    if k not in (source_field, target_field)
                }
                G.add_edge(src, tgt, **attrs)

        return G

    def nodes_and_links_to_networkx_digraph(
        nodes_and_links: dict, ctx: dict = None
    ) -> nx.DiGraph:
        """
        Convert nodes_and_links dict to NetworkX directed graph.

        Same as nodes_and_links_to_networkx_graph but returns DiGraph.

        Examples
        --------
        >>> data = {'nodes': [{'id': '0'}, {'id': '1'}], 'links': [{'source': '0', 'target': '1'}]}
        >>> G = nodes_and_links_to_networkx_digraph(data)
        >>> G.is_directed()
        True
        """
        # Reuse the graph conversion logic but create DiGraph
        G_undirected = nodes_and_links_to_networkx_graph(nodes_and_links, ctx)
        return nx.DiGraph(G_undirected)


# ============================================================================
# Register transformations
# ============================================================================

if _has_networkx:
    with suppress(ImportError, ModuleNotFoundError):
        # NetworkX Graph (undirected) conversions
        register_transformation('networkx_graph', 'edgelist', cost=0.3)(
            networkx_to_edgelist
        )
        register_transformation('edgelist', 'networkx_graph', cost=0.3)(
            edgelist_to_networkx_graph
        )
        register_transformation('networkx_graph', 'nodes_and_links', cost=0.3)(
            networkx_to_nodes_and_links
        )
        register_transformation('nodes_and_links', 'networkx_graph', cost=0.3)(
            nodes_and_links_to_networkx_graph
        )

        # NetworkX DiGraph (directed) conversions
        register_transformation('networkx_digraph', 'edgelist', cost=0.3)(
            networkx_to_edgelist
        )
        register_transformation('edgelist', 'networkx_digraph', cost=0.3)(
            edgelist_to_networkx_digraph
        )
        register_transformation('networkx_digraph', 'nodes_and_links', cost=0.3)(
            networkx_to_nodes_and_links
        )
        register_transformation('nodes_and_links', 'networkx_digraph', cost=0.3)(
            nodes_and_links_to_networkx_digraph
        )
