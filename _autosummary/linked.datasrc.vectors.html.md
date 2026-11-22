# linked.datasrc.vectors

Graph construction from high-dimensional vector data.

This module provides tools to convert point/vector data into graph data (links/edges)
suitable for network analysis and visualization. Focus is on sparse graph construction
from large datasets (50K-500K points) with high dimensionality (up to 3000+ dimensions).

### Functions

| [`adaptive_knn_graph`](#linked.datasrc.vectors.adaptive_knn_graph)(vectors, \*[, ...])             | Construct k-NN graph with adaptive distance scaling (UMAP-style).   |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [`epsilon_graph`](#linked.datasrc.vectors.epsilon_graph)(vectors, \*, radius[, metric, ...])  | Construct an epsilon-neighborhood graph.                            |
| [`knn_graph`](#linked.datasrc.vectors.knn_graph)(vectors, \*[, n_neighbors, metric, ...]) | Construct a k-nearest neighbors graph from vectors.                 |
| [`mutual_knn_graph`](#linked.datasrc.vectors.mutual_knn_graph)(vectors, \*[, n_neighbors, ...])  | Construct a mutual k-nearest neighbors graph.                       |
| [`random_graph`](#linked.datasrc.vectors.random_graph)(n_nodes, \*[, model, p, m, k, ...])   | Generate a random graph according to classical random graph models. |

### Classes

| [`AdaptiveKNNGraphEstimator`](#linked.datasrc.vectors.AdaptiveKNNGraphEstimator)([n_neighbors, ...])   | Sklearn-style estimator for adaptive k-NN graph construction.   |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| [`KNNGraphEstimator`](#linked.datasrc.vectors.KNNGraphEstimator)([n_neighbors, metric, ...])   | Sklearn-style estimator for k-NN graph construction.            |
| [`MutualKNNGraphEstimator`](#linked.datasrc.vectors.MutualKNNGraphEstimator)([n_neighbors, ...])     | Sklearn-style estimator for mutual k-NN graph construction.     |

### *class* linked.datasrc.vectors.AdaptiveKNNGraphEstimator(n_neighbors=15, metric='euclidean', local_connectivity=1, bandwidth=1.0, approximate=True, n_jobs=-1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Sklearn-style estimator for adaptive k-NN graph construction.

### Examples

```pycon
>>> estimator = AdaptiveKNNGraphEstimator(n_neighbors=5)
>>> vectors = np.random.rand(25, 6)
>>> graph = estimator.fit_transform(vectors)
>>> graph.shape[1] == 3
True
```

#### fit_transform(X)

Fit and transform in one step.

* **Return type:**
  `ndarray`

### *class* linked.datasrc.vectors.KNNGraphEstimator(n_neighbors=15, metric='euclidean', mode='distance', include_self=False, approximate=True, n_jobs=-1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Sklearn-style estimator for k-NN graph construction.

### Examples

```pycon
>>> estimator = KNNGraphEstimator(n_neighbors=5)
>>> vectors = np.random.rand(50, 10)
>>> graph = estimator.fit_transform(vectors)
>>> graph.shape[1] in [2, 3]
True
```

#### fit_transform(X)

Fit and transform in one step.

* **Return type:**
  `ndarray`

### *class* linked.datasrc.vectors.MutualKNNGraphEstimator(n_neighbors=15, metric='euclidean', mode='distance', approximate=True, ensure_connectivity='mst', n_jobs=-1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Sklearn-style estimator for mutual k-NN graph construction.

### Examples

```pycon
>>> estimator = MutualKNNGraphEstimator(n_neighbors=5)
>>> vectors = np.random.rand(30, 8)
>>> graph = estimator.fit_transform(vectors)
>>> len(graph) >= 0
True
```

#### fit_transform(X)

Fit and transform in one step.

* **Return type:**
  `ndarray`

### linked.datasrc.vectors.adaptive_knn_graph(vectors, , n_neighbors=15, metric='euclidean', local_connectivity=1, bandwidth=1.0, approximate=True, n_jobs=-1)

Construct k-NN graph with adaptive distance scaling (UMAP-style).

Uses local density estimation to adaptively scale distances, ensuring each
point has at least ‘local_connectivity’ strong connections. This helps
preserve local structure in varying-density regions.

* **Parameters:**
  * **vectors** (`ndarray`) – Input data of shape (n_samples, n_features)
  * **n_neighbors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of neighbors for each sample
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Distance metric
  * **local_connectivity** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum number of strong local connections per point
  * **bandwidth** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Gaussian kernel bandwidth for weight computation
  * **approximate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Use approximate nearest neighbors
  * **n_jobs** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of parallel jobs
* **Return type:**
  `ndarray`
* **Returns:**
  Graph as numpy array of shape (n_edges, 3) with adaptive weights

### Examples

```pycon
>>> vectors = np.random.rand(40, 8)
>>> graph = adaptive_knn_graph(vectors, n_neighbors=5)
>>> graph.shape[1] == 3
True
```

### linked.datasrc.vectors.epsilon_graph(vectors, , radius, metric='euclidean', mode='distance', approximate=False, n_jobs=-1)

Construct an epsilon-neighborhood graph.

Connect all pairs of points within distance radius of each other.

* **Parameters:**
  * **vectors** (`ndarray`) – Input data of shape (n_samples, n_features)
  * **radius** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Maximum distance for neighborhood
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Distance metric
  * **mode** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'connectivity'`, `'distance'`]) – ‘connectivity’ returns binary edges, ‘distance’ returns weighted edges
  * **approximate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Use approximate methods (not well-supported for radius queries)
  * **n_jobs** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of parallel jobs
* **Return type:**
  `ndarray`
* **Returns:**
  Graph as numpy array of shape (n_edges, 2) or (n_edges, 3)

### Examples

```pycon
>>> vectors = np.random.rand(30, 3)
>>> graph = epsilon_graph(vectors, radius=0.5)
>>> len(graph) >= 0
True
```

### linked.datasrc.vectors.knn_graph(vectors, , n_neighbors=15, metric='euclidean', mode='distance', include_self=False, approximate=True, n_jobs=-1)

Construct a k-nearest neighbors graph from vectors.

Each point is connected to its k nearest neighbors. Returns sparse edge list
with optional distance weights.

* **Parameters:**
  * **vectors** (`ndarray`) – Input data of shape (n_samples, n_features)
  * **n_neighbors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of neighbors for each sample
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Distance metric (euclidean, manhattan, cosine, etc.)
  * **mode** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'connectivity'`, `'distance'`]) – ‘connectivity’ returns binary edges, ‘distance’ returns weighted edges
  * **include_self** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to include self-loops
  * **approximate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Use approximate nearest neighbors for speed (requires pynndescent)
  * **n_jobs** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of parallel jobs (-1 for all cores)
* **Return type:**
  `ndarray`
* **Returns:**
  Graph as numpy array of shape (n_edges, 2) or (n_edges, 3).
  Columns are [source, target] or [source, target, weight].

### Examples

```pycon
>>> vectors = np.random.rand(100, 10)
>>> graph = knn_graph(vectors, n_neighbors=5)
>>> graph.shape[1] == 3  # source, target, weight
True
```

### linked.datasrc.vectors.mutual_knn_graph(vectors, , n_neighbors=15, metric='euclidean', mode='distance', approximate=True, ensure_connectivity='mst', n_jobs=-1)

Construct a mutual k-nearest neighbors graph.

An edge exists between points i and j only if i is among j’s k nearest neighbors
AND j is among i’s k nearest neighbors. This symmetrization reduces hub effects
and improves local structure preservation in high dimensions.

* **Parameters:**
  * **vectors** (`ndarray`) – Input data of shape (n_samples, n_features)
  * **n_neighbors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of neighbors for each sample
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Distance metric
  * **mode** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'connectivity'`, `'distance'`]) – ‘connectivity’ returns binary edges, ‘distance’ returns weighted edges
  * **approximate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Use approximate nearest neighbors for speed
  * **ensure_connectivity** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'none'`, `'mst'`, `'nn'`]) – 

    Method to ensure graph connectivity
    - ’none’: May result in disconnected components
    - ’mst’: Add minimum spanning tree edges
    - ’nn’: Connect isolated vertices to their nearest neighbor
  * **n_jobs** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of parallel jobs
* **Return type:**
  `ndarray`
* **Returns:**
  Graph as numpy array of shape (n_edges, 2) or (n_edges, 3)

### Examples

```pycon
>>> vectors = np.random.rand(50, 5)
>>> graph = mutual_knn_graph(vectors, n_neighbors=5)
>>> len(graph) > 0
True
```

### linked.datasrc.vectors.random_graph(n_nodes, , model='erdos_renyi', p=None, m=None, k=None, weighted=False, seed=None)

Generate a random graph according to classical random graph models.

* **Parameters:**
  * **n_nodes** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of nodes in the graph
  * **model** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'erdos_renyi'`, `'barabasi_albert'`, `'watts_strogatz'`]) – 

    Random graph model to use
    - ’erdos_renyi’: Each edge exists with probability p
    - ’barabasi_albert’: Preferential attachment, add m edges per new node
    - ’watts_strogatz’: Small-world, start with k-regular lattice, rewire with prob p
  * **p** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Edge probability (erdos_renyi, watts_strogatz)
  * **m** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Edges per node (barabasi_albert)
  * **k** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Initial degree (watts_strogatz)
  * **weighted** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, assign random weights to edges
  * **seed** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Random seed for reproducibility
* **Return type:**
  `ndarray`
* **Returns:**
  Graph as numpy array of shape (n_edges, 2) or (n_edges, 3)

### Examples

```pycon
>>> graph = random_graph(20, model="erdos_renyi", p=0.3, seed=42)
>>> len(graph) > 0
True
```
