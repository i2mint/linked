"""
Adjacency matrix and adjacency list graph representations.

Adjacency formats are classical graph representations used in algorithms and analysis.

Kinds defined here
------------------
- 'adjacency_matrix': Dense numpy array (n_nodes, n_nodes)
- 'sparse_adjacency': scipy sparse matrix (CSR, CSC, COO, etc.)
- 'adjacency_list': Dict mapping node index to list of neighbor indices

Conversions
-----------
- adjacency matrix <-> edgelist
- sparse adjacency <-> edgelist
- adjacency list <-> edgelist
- adjacency matrix <-> sparse adjacency
"""

from contextlib import suppress
import numpy as np

# Try to import scipy - only register if available
_has_scipy = False
with suppress(ImportError, ModuleNotFoundError):
    from scipy import sparse

    _has_scipy = True

# Only register if cast is available
with suppress(ImportError, ModuleNotFoundError):
    from linked.cast import register_kind, register_transformation


# ============================================================================
# Kind predicates
# ============================================================================


def _is_adjacency_list(obj) -> bool:
    """Check if object is an adjacency list (dict of lists)."""
    if not isinstance(obj, dict):
        return False
    # Check if values are lists/iterables of integers
    for val in obj.values():
        if not hasattr(val, '__iter__'):
            return False
        break  # Just check first one
    return True


if _has_scipy:

    def _is_sparse_adjacency(obj) -> bool:
        """Check if object is a scipy sparse matrix."""
        return sparse.issparse(obj)

    # Register kinds
    with suppress(ImportError, ModuleNotFoundError):
        register_kind('sparse_adjacency', isa=_is_sparse_adjacency)


with suppress(ImportError, ModuleNotFoundError):
    register_kind('adjacency_list', isa=_is_adjacency_list)


# ============================================================================
# Adjacency matrix <-> edge list conversions
# ============================================================================


def adjacency_matrix_to_edgelist(
    adj_matrix: np.ndarray, ctx: dict = None
) -> np.ndarray:
    """
    Convert adjacency matrix to edge list.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Square adjacency matrix of shape (n_nodes, n_nodes)
    ctx : dict, optional
        Context with optional keys:
        - 'directed': whether graph is directed (default: auto-detect from symmetry)
        - 'include_weights': whether to include weights (default: True if non-binary)
        - 'threshold': minimum weight to include edge (default: 0)

    Returns
    -------
    np.ndarray
        Edge list array

    Examples
    --------
    >>> adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> edgelist = adjacency_matrix_to_edgelist(adj)
    >>> len(edgelist) > 0
    True
    """
    ctx = ctx or {}
    threshold = ctx.get('threshold', 0)

    # Auto-detect if directed
    directed = ctx.get('directed', None)
    if directed is None:
        directed = not np.allclose(adj_matrix, adj_matrix.T)

    # Auto-detect if we should include weights
    include_weights = ctx.get('include_weights', None)
    if include_weights is None:
        # Include weights if matrix has values other than 0 and 1
        unique_vals = np.unique(adj_matrix[adj_matrix > threshold])
        include_weights = len(unique_vals) > 1 or (
            len(unique_vals) == 1 and unique_vals[0] != 1
        )

    # Get edges
    if directed:
        # For directed graphs, include all edges
        i_idx, j_idx = np.where(adj_matrix > threshold)
    else:
        # For undirected graphs, only include upper triangle to avoid duplicates
        i_idx, j_idx = np.where(np.triu(adj_matrix, k=1) > threshold)

    if len(i_idx) == 0:
        return np.empty((0, 3 if include_weights else 2))

    if include_weights:
        weights = adj_matrix[i_idx, j_idx]
        return np.column_stack([i_idx, j_idx, weights])
    else:
        return np.column_stack([i_idx, j_idx])


def edgelist_to_adjacency_matrix(edgelist: np.ndarray, ctx: dict = None) -> np.ndarray:
    """
    Convert edge list to adjacency matrix.

    Parameters
    ----------
    edgelist : np.ndarray
        Edge list array
    ctx : dict, optional
        Context with optional keys:
        - 'n_nodes': number of nodes (default: max index + 1)
        - 'directed': whether graph is directed (default: False)
        - 'dtype': data type for matrix (default: float if weighted, int otherwise)

    Returns
    -------
    np.ndarray
        Adjacency matrix

    Examples
    --------
    >>> edgelist = np.array([[0, 1], [1, 2]])
    >>> adj = edgelist_to_adjacency_matrix(edgelist)
    >>> adj.shape[0] == adj.shape[1]
    True
    """
    ctx = ctx or {}
    directed = ctx.get('directed', False)

    if len(edgelist) == 0:
        n_nodes = ctx.get('n_nodes', 0)
        return np.zeros((n_nodes, n_nodes))

    # Determine number of nodes
    n_nodes = ctx.get('n_nodes', None)
    if n_nodes is None:
        n_nodes = int(edgelist[:, :2].max()) + 1

    # Determine dtype
    has_weights = edgelist.shape[1] >= 3
    dtype = ctx.get('dtype', None)
    if dtype is None:
        dtype = float if has_weights else int

    # Initialize matrix
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=dtype)

    # Fill matrix
    for edge in edgelist:
        i = int(edge[0])
        j = int(edge[1])
        weight = edge[2] if has_weights else 1

        adj_matrix[i, j] = weight
        if not directed:
            adj_matrix[j, i] = weight

    return adj_matrix


# ============================================================================
# Sparse adjacency <-> edge list conversions
# ============================================================================

if _has_scipy:

    def sparse_adjacency_to_edgelist(sparse_adj, ctx: dict = None) -> np.ndarray:
        """
        Convert sparse adjacency matrix to edge list.

        Parameters
        ----------
        sparse_adj : scipy.sparse matrix
            Sparse adjacency matrix
        ctx : dict, optional
            Context (same as adjacency_matrix_to_edgelist)

        Returns
        -------
        np.ndarray
            Edge list array

        Examples
        --------
        >>> from scipy import sparse
        >>> adj = sparse.csr_matrix(([1, 1], ([0, 1], [1, 2])), shape=(3, 3))
        >>> edgelist = sparse_adjacency_to_edgelist(adj)
        >>> len(edgelist) > 0
        True
        """
        ctx = ctx or {}

        # Convert to COO format for easy access to edges
        coo = sparse_adj.tocoo()

        # Auto-detect directed
        directed = ctx.get('directed', None)
        if directed is None:
            # Check if symmetric
            directed = not (coo != coo.T).nnz == 0

        # Build edge list
        if directed:
            edges = np.column_stack([coo.row, coo.col, coo.data])
        else:
            # For undirected, only keep upper triangle
            mask = coo.row <= coo.col
            edges = np.column_stack([coo.row[mask], coo.col[mask], coo.data[mask]])

        # Check if we should include weights
        include_weights = ctx.get('include_weights', None)
        if include_weights is None:
            unique_vals = np.unique(edges[:, 2])
            include_weights = len(unique_vals) > 1 or (
                len(unique_vals) == 1 and unique_vals[0] != 1
            )

        if not include_weights:
            edges = edges[:, :2]

        return edges

    def edgelist_to_sparse_adjacency(edgelist: np.ndarray, ctx: dict = None):
        """
        Convert edge list to sparse adjacency matrix.

        Parameters
        ----------
        edgelist : np.ndarray
            Edge list array
        ctx : dict, optional
            Context with optional keys:
            - 'n_nodes': number of nodes (default: max index + 1)
            - 'directed': whether graph is directed (default: False)
            - 'format': sparse matrix format - 'csr', 'csc', 'coo' (default: 'csr')

        Returns
        -------
        scipy.sparse matrix
            Sparse adjacency matrix

        Examples
        --------
        >>> edgelist = np.array([[0, 1, 0.5], [1, 2, 0.8]])
        >>> sparse_adj = edgelist_to_sparse_adjacency(edgelist)
        >>> sparse_adj.shape[0] == sparse_adj.shape[1]
        True
        """
        ctx = ctx or {}
        directed = ctx.get('directed', False)
        format = ctx.get('format', 'csr')

        if len(edgelist) == 0:
            n_nodes = ctx.get('n_nodes', 0)
            return sparse.csr_matrix((n_nodes, n_nodes))

        # Determine number of nodes
        n_nodes = ctx.get('n_nodes', None)
        if n_nodes is None:
            n_nodes = int(edgelist[:, :2].max()) + 1

        # Extract edges
        has_weights = edgelist.shape[1] >= 3

        if directed:
            # Just use edges as-is
            row = edgelist[:, 0].astype(int)
            col = edgelist[:, 1].astype(int)
            data = edgelist[:, 2] if has_weights else np.ones(len(edgelist))
        else:
            # For undirected, add both directions
            row = np.concatenate([edgelist[:, 0], edgelist[:, 1]]).astype(int)
            col = np.concatenate([edgelist[:, 1], edgelist[:, 0]]).astype(int)
            if has_weights:
                data = np.concatenate([edgelist[:, 2], edgelist[:, 2]])
            else:
                data = np.ones(len(row))

        # Create sparse matrix
        sparse_adj = sparse.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        # Convert to desired format
        if format == 'csr':
            return sparse_adj.tocsr()
        elif format == 'csc':
            return sparse_adj.tocsc()
        else:
            return sparse_adj


# ============================================================================
# Adjacency list <-> edge list conversions
# ============================================================================


def adjacency_list_to_edgelist(adj_list: dict, ctx: dict = None) -> np.ndarray:
    """
    Convert adjacency list to edge list.

    Parameters
    ----------
    adj_list : dict
        Dict mapping node index to list of neighbor indices
    ctx : dict, optional
        Context with optional keys:
        - 'include_weights': whether edges have weights (default: False)
        - 'directed': whether graph is directed (default: True for adj lists)

    Returns
    -------
    np.ndarray
        Edge list array

    Examples
    --------
    >>> adj_list = {0: [1, 2], 1: [2], 2: []}
    >>> edgelist = adjacency_list_to_edgelist(adj_list)
    >>> len(edgelist) == 3
    True
    """
    ctx = ctx or {}
    include_weights = ctx.get('include_weights', False)

    edges = []
    for src, neighbors in adj_list.items():
        if isinstance(neighbors, dict):
            # Neighbors are {node: weight} dict
            for tgt, weight in neighbors.items():
                edges.append([src, tgt, weight])
            include_weights = True
        else:
            # Neighbors are a list/iterable
            for tgt in neighbors:
                if include_weights:
                    edges.append([src, tgt, 1.0])
                else:
                    edges.append([src, tgt])

    if not edges:
        return np.empty((0, 3 if include_weights else 2))

    return np.array(edges)


def edgelist_to_adjacency_list(edgelist: np.ndarray, ctx: dict = None) -> dict:
    """
    Convert edge list to adjacency list.

    Parameters
    ----------
    edgelist : np.ndarray
        Edge list array
    ctx : dict, optional
        Context with optional keys:
        - 'directed': whether graph is directed (default: False)
        - 'include_weights': store weights in adjacency list (default: auto-detect)

    Returns
    -------
    dict
        Adjacency list dict

    Examples
    --------
    >>> edgelist = np.array([[0, 1], [1, 2], [0, 2]])
    >>> adj_list = edgelist_to_adjacency_list(edgelist)
    >>> len(adj_list[0]) == 2
    True
    """
    ctx = ctx or {}
    directed = ctx.get('directed', False)
    include_weights = ctx.get('include_weights', None)

    if len(edgelist) == 0:
        return {}

    has_weights = edgelist.shape[1] >= 3
    if include_weights is None:
        include_weights = has_weights

    # Initialize adjacency list
    adj_list = {}

    # Get all node indices
    all_nodes = np.unique(edgelist[:, :2].astype(int))
    for node in all_nodes:
        adj_list[int(node)] = {} if include_weights else []

    # Add edges
    for edge in edgelist:
        src = int(edge[0])
        tgt = int(edge[1])

        if src not in adj_list:
            adj_list[src] = {} if include_weights else []
        if tgt not in adj_list:
            adj_list[tgt] = {} if include_weights else []

        if include_weights:
            weight = float(edge[2]) if has_weights else 1.0
            adj_list[src][tgt] = weight
            if not directed:
                adj_list[tgt][src] = weight
        else:
            if tgt not in adj_list[src]:
                adj_list[src].append(tgt)
            if not directed and src not in adj_list[tgt]:
                adj_list[tgt].append(src)

    return adj_list


# ============================================================================
# Adjacency matrix <-> sparse adjacency conversions
# ============================================================================

if _has_scipy:

    def adjacency_matrix_to_sparse_adjacency(adj_matrix: np.ndarray, ctx: dict = None):
        """Convert dense adjacency matrix to sparse format."""
        ctx = ctx or {}
        format = ctx.get('format', 'csr')

        sparse_adj = sparse.coo_matrix(adj_matrix)

        if format == 'csr':
            return sparse_adj.tocsr()
        elif format == 'csc':
            return sparse_adj.tocsc()
        else:
            return sparse_adj

    def sparse_adjacency_to_adjacency_matrix(
        sparse_adj, ctx: dict = None
    ) -> np.ndarray:
        """Convert sparse adjacency matrix to dense format."""
        return sparse_adj.toarray()


# ============================================================================
# Register transformations
# ============================================================================

with suppress(ImportError, ModuleNotFoundError):
    # Adjacency matrix conversions
    register_transformation('adjacency_matrix', 'edgelist', cost=0.5)(
        adjacency_matrix_to_edgelist
    )
    register_transformation('edgelist', 'adjacency_matrix', cost=0.5)(
        edgelist_to_adjacency_matrix
    )

    # Adjacency list conversions
    register_transformation('adjacency_list', 'edgelist', cost=0.4)(
        adjacency_list_to_edgelist
    )
    register_transformation('edgelist', 'adjacency_list', cost=0.4)(
        edgelist_to_adjacency_list
    )

if _has_scipy:
    with suppress(ImportError, ModuleNotFoundError):
        # Sparse adjacency conversions
        register_transformation('sparse_adjacency', 'edgelist', cost=0.4)(
            sparse_adjacency_to_edgelist
        )
        register_transformation('edgelist', 'sparse_adjacency', cost=0.4)(
            edgelist_to_sparse_adjacency
        )

        # Dense <-> sparse conversions
        register_transformation('adjacency_matrix', 'sparse_adjacency', cost=0.2)(
            adjacency_matrix_to_sparse_adjacency
        )
        register_transformation('sparse_adjacency', 'adjacency_matrix', cost=0.2)(
            sparse_adjacency_to_adjacency_matrix
        )
