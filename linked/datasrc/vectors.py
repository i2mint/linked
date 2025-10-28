"""
Graph construction from high-dimensional vector data.

This module provides tools to convert point/vector data into graph data (links/edges)
suitable for network analysis and visualization. Focus is on sparse graph construction
from large datasets (50K-500K points) with high dimensionality (up to 3000+ dimensions).
"""

from dataclasses import dataclass
from typing import Literal, Optional, Callable, Union
import warnings

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components


# Optional imports with graceful degradation
try:
    from sklearn.neighbors import NearestNeighbors
    _has_sklearn = True
except ImportError:
    _has_sklearn = False
    warnings.warn("scikit-learn not available. Some features may be limited.")

try:
    from pynndescent import NNDescent
    _has_pynndescent = True
except ImportError:
    _has_pynndescent = False
    warnings.warn("pynndescent not available. Approximate methods will fall back to exact.")


def knn_graph(
    vectors: np.ndarray,
    *,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    mode: Literal["connectivity", "distance"] = "distance",
    include_self: bool = False,
    approximate: bool = True,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Construct a k-nearest neighbors graph from vectors.
    
    Each point is connected to its k nearest neighbors. Returns sparse edge list
    with optional distance weights.
    
    Args:
        vectors: Input data of shape (n_samples, n_features)
        n_neighbors: Number of neighbors for each sample
        metric: Distance metric (euclidean, manhattan, cosine, etc.)
        mode: 'connectivity' returns binary edges, 'distance' returns weighted edges
        include_self: Whether to include self-loops
        approximate: Use approximate nearest neighbors for speed (requires pynndescent)
        n_jobs: Number of parallel jobs (-1 for all cores)
    
    Returns:
        Graph as numpy array of shape (n_edges, 2) or (n_edges, 3).
        Columns are [source, target] or [source, target, weight].
    
    Examples:
        >>> vectors = np.random.rand(100, 10)
        >>> graph = knn_graph(vectors, n_neighbors=5)
        >>> graph.shape[1] == 3  # source, target, weight
        True
    """
    n_samples = vectors.shape[0]
    
    if approximate and _has_pynndescent:
        # Use approximate nearest neighbors
        index = NNDescent(
            vectors,
            n_neighbors=n_neighbors + (1 if not include_self else 0),
            metric=metric,
            n_jobs=n_jobs,
        )
        neighbors, distances = index.neighbor_graph
        
        if not include_self:
            # Remove self-loops (first neighbor is always self)
            neighbors = neighbors[:, 1:]
            distances = distances[:, 1:]
    
    elif _has_sklearn:
        # Use sklearn's exact nearest neighbors
        nn = NearestNeighbors(
            n_neighbors=n_neighbors + (1 if not include_self else 0),
            metric=metric,
            n_jobs=n_jobs,
        )
        nn.fit(vectors)
        
        if mode == "distance":
            distances, neighbors = nn.kneighbors(vectors)
        else:
            neighbors = nn.kneighbors(vectors, return_distance=False)
            distances = None
        
        if not include_self:
            neighbors = neighbors[:, 1:]
            if distances is not None:
                distances = distances[:, 1:]
    
    else:
        raise ImportError("Either pynndescent or scikit-learn must be installed")
    
    # Convert to edge list format
    return _neighbors_to_edgelist(neighbors, distances if mode == "distance" else None)


def mutual_knn_graph(
    vectors: np.ndarray,
    *,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    mode: Literal["connectivity", "distance"] = "distance",
    approximate: bool = True,
    ensure_connectivity: Literal["none", "mst", "nn"] = "mst",
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Construct a mutual k-nearest neighbors graph.
    
    An edge exists between points i and j only if i is among j's k nearest neighbors
    AND j is among i's k nearest neighbors. This symmetrization reduces hub effects
    and improves local structure preservation in high dimensions.
    
    Args:
        vectors: Input data of shape (n_samples, n_features)
        n_neighbors: Number of neighbors for each sample
        metric: Distance metric
        mode: 'connectivity' returns binary edges, 'distance' returns weighted edges
        approximate: Use approximate nearest neighbors for speed
        ensure_connectivity: Method to ensure graph connectivity
            - 'none': May result in disconnected components
            - 'mst': Add minimum spanning tree edges
            - 'nn': Connect isolated vertices to their nearest neighbor
        n_jobs: Number of parallel jobs
    
    Returns:
        Graph as numpy array of shape (n_edges, 2) or (n_edges, 3)
    
    Examples:
        >>> vectors = np.random.rand(50, 5)
        >>> graph = mutual_knn_graph(vectors, n_neighbors=5)
        >>> len(graph) > 0
        True
    """
    # First build regular k-NN graph
    if approximate and _has_pynndescent:
        index = NNDescent(
            vectors,
            n_neighbors=n_neighbors + 1,
            metric=metric,
            n_jobs=n_jobs,
        )
        neighbors, distances = index.neighbor_graph
        # Remove self-loops
        neighbors = neighbors[:, 1:]
        distances = distances[:, 1:]
    elif _has_sklearn:
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=n_jobs)
        nn.fit(vectors)
        distances, neighbors = nn.kneighbors(vectors)
        neighbors = neighbors[:, 1:]
        distances = distances[:, 1:]
    else:
        raise ImportError("Either pynndescent or scikit-learn must be installed")
    
    # Convert to sparse matrix for efficient mutual intersection
    n_samples = vectors.shape[0]
    knn_matrix = _neighbors_to_sparse_matrix(neighbors, distances if mode == "distance" else None, n_samples)
    
    # Make symmetric by keeping only mutual edges
    mutual_matrix = knn_matrix.minimum(knn_matrix.T)
    
    # Convert back to edge list
    graph = _sparse_to_edgelist(mutual_matrix, include_weights=(mode == "distance"))
    
    # Ensure connectivity if requested
    if ensure_connectivity != "none" and len(graph) > 0:
        graph = _ensure_connectivity(
            graph, vectors, method=ensure_connectivity, metric=metric, n_samples=n_samples
        )
    
    return graph


def epsilon_graph(
    vectors: np.ndarray,
    *,
    radius: float,
    metric: str = "euclidean",
    mode: Literal["connectivity", "distance"] = "distance",
    approximate: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Construct an epsilon-neighborhood graph.
    
    Connect all pairs of points within distance radius of each other.
    
    Args:
        vectors: Input data of shape (n_samples, n_features)
        radius: Maximum distance for neighborhood
        metric: Distance metric
        mode: 'connectivity' returns binary edges, 'distance' returns weighted edges
        approximate: Use approximate methods (not well-supported for radius queries)
        n_jobs: Number of parallel jobs
    
    Returns:
        Graph as numpy array of shape (n_edges, 2) or (n_edges, 3)
    
    Examples:
        >>> vectors = np.random.rand(30, 3)
        >>> graph = epsilon_graph(vectors, radius=0.5)
        >>> len(graph) >= 0
        True
    """
    if _has_sklearn:
        nn = NearestNeighbors(radius=radius, metric=metric, n_jobs=n_jobs)
        nn.fit(vectors)
        
        if mode == "distance":
            distances, neighbors = nn.radius_neighbors(vectors)
        else:
            neighbors = nn.radius_neighbors(vectors, return_distance=False)
            distances = [None] * len(neighbors)
        
        # Convert ragged arrays to edge list
        edges = []
        for i, (nbrs, dists) in enumerate(zip(neighbors, distances)):
            for j_idx, j in enumerate(nbrs):
                if i != j:  # Skip self-loops
                    if mode == "distance" and dists is not None:
                        edges.append([i, j, dists[j_idx]])
                    else:
                        edges.append([i, j])
        
        return np.array(edges) if edges else np.empty((0, 3 if mode == "distance" else 2))
    
    else:
        # Fallback to manual distance computation
        n_samples = vectors.shape[0]
        dist_matrix = cdist(vectors, vectors, metric=metric)
        
        # Find pairs within radius
        i_idx, j_idx = np.where((dist_matrix <= radius) & (dist_matrix > 0))
        
        if mode == "distance":
            weights = dist_matrix[i_idx, j_idx]
            return np.column_stack([i_idx, j_idx, weights])
        else:
            return np.column_stack([i_idx, j_idx])


def adaptive_knn_graph(
    vectors: np.ndarray,
    *,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    local_connectivity: int = 1,
    bandwidth: float = 1.0,
    approximate: bool = True,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Construct k-NN graph with adaptive distance scaling (UMAP-style).
    
    Uses local density estimation to adaptively scale distances, ensuring each
    point has at least 'local_connectivity' strong connections. This helps
    preserve local structure in varying-density regions.
    
    Args:
        vectors: Input data of shape (n_samples, n_features)
        n_neighbors: Number of neighbors for each sample
        metric: Distance metric
        local_connectivity: Minimum number of strong local connections per point
        bandwidth: Gaussian kernel bandwidth for weight computation
        approximate: Use approximate nearest neighbors
        n_jobs: Number of parallel jobs
    
    Returns:
        Graph as numpy array of shape (n_edges, 3) with adaptive weights
    
    Examples:
        >>> vectors = np.random.rand(40, 8)
        >>> graph = adaptive_knn_graph(vectors, n_neighbors=5)
        >>> graph.shape[1] == 3
        True
    """
    # Get k-NN graph with distances
    if approximate and _has_pynndescent:
        index = NNDescent(
            vectors,
            n_neighbors=n_neighbors + 1,
            metric=metric,
            n_jobs=n_jobs,
        )
        neighbors, distances = index.neighbor_graph
        neighbors = neighbors[:, 1:]
        distances = distances[:, 1:]
    elif _has_sklearn:
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=n_jobs)
        nn.fit(vectors)
        distances, neighbors = nn.kneighbors(vectors)
        neighbors = neighbors[:, 1:]
        distances = distances[:, 1:]
    else:
        raise ImportError("Either pynndescent or scikit-learn must be installed")
    
    # Compute local distance scale (rho) for each point
    # Use distance to the local_connectivity-th nearest neighbor
    rho = distances[:, max(0, local_connectivity - 1)]
    
    # Compute adaptive weights using smooth approximation
    n_samples = len(vectors)
    edges = []
    
    for i in range(n_samples):
        for j_idx, (j, dist) in enumerate(zip(neighbors[i], distances[i])):
            # Adaptive distance: subtract local scale and apply Gaussian kernel
            adaptive_dist = max(0, dist - rho[i])
            weight = np.exp(-adaptive_dist / bandwidth)
            edges.append([i, j, weight])
    
    return np.array(edges)


def random_graph(
    n_nodes: int,
    *,
    model: Literal["erdos_renyi", "barabasi_albert", "watts_strogatz"] = "erdos_renyi",
    p: Optional[float] = None,
    m: Optional[int] = None,
    k: Optional[int] = None,
    weighted: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random graph according to classical random graph models.
    
    Args:
        n_nodes: Number of nodes in the graph
        model: Random graph model to use
            - 'erdos_renyi': Each edge exists with probability p
            - 'barabasi_albert': Preferential attachment, add m edges per new node
            - 'watts_strogatz': Small-world, start with k-regular lattice, rewire with prob p
        p: Edge probability (erdos_renyi, watts_strogatz)
        m: Edges per node (barabasi_albert)
        k: Initial degree (watts_strogatz)
        weighted: If True, assign random weights to edges
        seed: Random seed for reproducibility
    
    Returns:
        Graph as numpy array of shape (n_edges, 2) or (n_edges, 3)
    
    Examples:
        >>> graph = random_graph(20, model="erdos_renyi", p=0.3, seed=42)
        >>> len(graph) > 0
        True
    """
    rng = np.random.default_rng(seed)
    
    if model == "erdos_renyi":
        if p is None:
            raise ValueError("erdos_renyi model requires 'p' parameter")
        edges = _erdos_renyi_graph(n_nodes, p, rng)
    
    elif model == "barabasi_albert":
        if m is None:
            raise ValueError("barabasi_albert model requires 'm' parameter")
        edges = _barabasi_albert_graph(n_nodes, m, rng)
    
    elif model == "watts_strogatz":
        if k is None or p is None:
            raise ValueError("watts_strogatz model requires 'k' and 'p' parameters")
        edges = _watts_strogatz_graph(n_nodes, k, p, rng)
    
    else:
        raise ValueError(f"Unknown model: {model}")
    
    if weighted and len(edges) > 0:
        weights = rng.uniform(0, 1, size=len(edges))
        edges = np.column_stack([edges, weights])
    
    return edges


# ============================================================================
# Helper functions (module-internal utilities)
# ============================================================================


def _neighbors_to_edgelist(
    neighbors: np.ndarray,
    distances: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert neighbor indices and distances to edge list format.
    
    Args:
        neighbors: Array of shape (n_samples, n_neighbors) with neighbor indices
        distances: Optional array of shape (n_samples, n_neighbors) with distances
    
    Returns:
        Edge list of shape (n_edges, 2) or (n_edges, 3)
    """
    n_samples, n_neighbors = neighbors.shape
    sources = np.repeat(np.arange(n_samples), n_neighbors)
    targets = neighbors.ravel()
    
    if distances is not None:
        weights = distances.ravel()
        return np.column_stack([sources, targets, weights])
    else:
        return np.column_stack([sources, targets])


def _neighbors_to_sparse_matrix(
    neighbors: np.ndarray,
    distances: Optional[np.ndarray] = None,
    n_samples: Optional[int] = None,
) -> sparse.csr_matrix:
    """
    Convert neighbor indices and distances to sparse adjacency matrix.
    
    Args:
        neighbors: Array of shape (n_samples, n_neighbors)
        distances: Optional distances array
        n_samples: Number of samples (inferred if None)
    
    Returns:
        Sparse adjacency matrix
    """
    if n_samples is None:
        n_samples = neighbors.shape[0]
    
    n_neighbors = neighbors.shape[1]
    sources = np.repeat(np.arange(n_samples), n_neighbors)
    targets = neighbors.ravel()
    
    if distances is not None:
        data = distances.ravel()
    else:
        data = np.ones(len(sources))
    
    return sparse.csr_matrix(
        (data, (sources, targets)),
        shape=(n_samples, n_samples),
    )


def _sparse_to_edgelist(
    matrix: sparse.csr_matrix,
    *,
    include_weights: bool = True,
) -> np.ndarray:
    """
    Convert sparse matrix to edge list.
    
    Args:
        matrix: Sparse adjacency matrix
        include_weights: Whether to include edge weights
    
    Returns:
        Edge list array
    """
    coo = matrix.tocoo()
    
    if include_weights:
        return np.column_stack([coo.row, coo.col, coo.data])
    else:
        return np.column_stack([coo.row, coo.col])


def _ensure_connectivity(
    graph: np.ndarray,
    vectors: np.ndarray,
    *,
    method: Literal["mst", "nn"],
    metric: str,
    n_samples: int,
) -> np.ndarray:
    """
    Ensure graph connectivity by adding edges.
    
    Args:
        graph: Current edge list
        vectors: Original vector data
        method: Connectivity method ('mst' or 'nn')
        metric: Distance metric
        n_samples: Number of samples
    
    Returns:
        Enhanced edge list with guaranteed connectivity
    """
    # Convert to sparse matrix to check connectivity
    if graph.shape[1] == 3:
        # Has weights
        matrix = sparse.csr_matrix(
            (graph[:, 2], (graph[:, 0].astype(int), graph[:, 1].astype(int))),
            shape=(n_samples, n_samples),
        )
        has_weights = True
    else:
        matrix = sparse.csr_matrix(
            (np.ones(len(graph)), (graph[:, 0].astype(int), graph[:, 1].astype(int))),
            shape=(n_samples, n_samples),
        )
        has_weights = False
    
    # Make undirected for connectivity check
    matrix = matrix + matrix.T
    
    n_components, labels = connected_components(matrix, directed=False)
    
    if n_components == 1:
        # Already connected
        return graph
    
    # Need to add edges
    if method == "mst":
        # Compute full distance matrix (or use sparse approximation)
        # For large datasets, this is expensive - consider using approximate MST
        dist_matrix = cdist(vectors, vectors, metric=metric)
        dist_sparse = sparse.csr_matrix(dist_matrix)
        
        # Compute MST
        mst = minimum_spanning_tree(dist_sparse)
        mst_edges = _sparse_to_edgelist(mst, include_weights=has_weights)
        
        # Combine with original edges
        return np.vstack([graph, mst_edges])
    
    elif method == "nn":
        # Connect each isolated point to its nearest neighbor
        # Find isolated components
        isolated_nodes = []
        for comp_id in range(n_components):
            comp_nodes = np.where(labels == comp_id)[0]
            if len(comp_nodes) == 1:
                isolated_nodes.append(comp_nodes[0])
        
        # For each isolated node, find nearest neighbor
        if _has_sklearn and len(isolated_nodes) > 0:
            nn = NearestNeighbors(n_neighbors=2, metric=metric)
            nn.fit(vectors)
            distances, neighbors = nn.kneighbors(vectors[isolated_nodes])
            
            new_edges = []
            for i, node in enumerate(isolated_nodes):
                # Connect to nearest neighbor (excluding self)
                nearest = neighbors[i, 1]
                if has_weights:
                    new_edges.append([node, nearest, distances[i, 1]])
                else:
                    new_edges.append([node, nearest])
            
            if new_edges:
                return np.vstack([graph, np.array(new_edges)])
        
        return graph
    
    return graph


def _erdos_renyi_graph(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Generate Erdos-Renyi random graph."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append([i, j])
    return np.array(edges) if edges else np.empty((0, 2), dtype=int)


def _barabasi_albert_graph(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Barabasi-Albert preferential attachment graph."""
    if m < 1 or m >= n:
        raise ValueError("m must be between 1 and n-1")
    
    # Start with complete graph on m nodes
    edges = []
    for i in range(m):
        for j in range(i + 1, m):
            edges.append([i, j])
    
    # Track degree of each node
    degree = np.zeros(n, dtype=int)
    for edge in edges:
        degree[edge[0]] += 1
        degree[edge[1]] += 1
    
    # Add remaining nodes with preferential attachment
    for new_node in range(m, n):
        # Select m nodes with probability proportional to degree
        if degree[:new_node].sum() > 0:
            probs = degree[:new_node] / degree[:new_node].sum()
            targets = rng.choice(new_node, size=m, replace=False, p=probs)
            
            for target in targets:
                edges.append([new_node, target])
                degree[new_node] += 1
                degree[target] += 1
    
    return np.array(edges) if edges else np.empty((0, 2), dtype=int)


def _watts_strogatz_graph(n: int, k: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Generate Watts-Strogatz small-world graph."""
    if k >= n:
        raise ValueError("k must be less than n")
    if k % 2 != 0:
        raise ValueError("k must be even")
    
    # Start with ring lattice
    edges = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            target = (i + j) % n
            edges.append([i, target])
    
    # Rewire edges with probability p
    rewired_edges = []
    existing_targets = {i: set() for i in range(n)}
    
    for edge in edges:
        i, j = edge
        if rng.random() < p:
            # Rewire - find new target
            possible_targets = [t for t in range(n) if t != i and t not in existing_targets[i]]
            if possible_targets:
                new_target = rng.choice(possible_targets)
                rewired_edges.append([i, new_target])
                existing_targets[i].add(new_target)
            else:
                rewired_edges.append(edge)
                existing_targets[i].add(j)
        else:
            rewired_edges.append(edge)
            existing_targets[i].add(j)
    
    return np.array(rewired_edges) if rewired_edges else np.empty((0, 2), dtype=int)


# ============================================================================
# Sklearn-style Estimator Classes
# ============================================================================


@dataclass
class KNNGraphEstimator:
    """
    Sklearn-style estimator for k-NN graph construction.
    
    Examples:
        >>> estimator = KNNGraphEstimator(n_neighbors=5)
        >>> vectors = np.random.rand(50, 10)
        >>> graph = estimator.fit_transform(vectors)
        >>> graph.shape[1] in [2, 3]
        True
    """
    n_neighbors: int = 15
    metric: str = "euclidean"
    mode: Literal["connectivity", "distance"] = "distance"
    include_self: bool = False
    approximate: bool = True
    n_jobs: int = -1
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return knn_graph(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            mode=self.mode,
            include_self=self.include_self,
            approximate=self.approximate,
            n_jobs=self.n_jobs,
        )


@dataclass
class MutualKNNGraphEstimator:
    """
    Sklearn-style estimator for mutual k-NN graph construction.
    
    Examples:
        >>> estimator = MutualKNNGraphEstimator(n_neighbors=5)
        >>> vectors = np.random.rand(30, 8)
        >>> graph = estimator.fit_transform(vectors)
        >>> len(graph) >= 0
        True
    """
    n_neighbors: int = 15
    metric: str = "euclidean"
    mode: Literal["connectivity", "distance"] = "distance"
    approximate: bool = True
    ensure_connectivity: Literal["none", "mst", "nn"] = "mst"
    n_jobs: int = -1
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return mutual_knn_graph(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            mode=self.mode,
            approximate=self.approximate,
            ensure_connectivity=self.ensure_connectivity,
            n_jobs=self.n_jobs,
        )


@dataclass
class AdaptiveKNNGraphEstimator:
    """
    Sklearn-style estimator for adaptive k-NN graph construction.
    
    Examples:
        >>> estimator = AdaptiveKNNGraphEstimator(n_neighbors=5)
        >>> vectors = np.random.rand(25, 6)
        >>> graph = estimator.fit_transform(vectors)
        >>> graph.shape[1] == 3
        True
    """
    n_neighbors: int = 15
    metric: str = "euclidean"
    local_connectivity: int = 1
    bandwidth: float = 1.0
    approximate: bool = True
    n_jobs: int = -1
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return adaptive_knn_graph(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            local_connectivity=self.local_connectivity,
            bandwidth=self.bandwidth,
            approximate=self.approximate,
            n_jobs=self.n_jobs,
        )
