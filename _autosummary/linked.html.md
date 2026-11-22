# linked

Create and transform graph data.

The linked package provides tools for creating and converting between various
graph data representations. It includes:

- Graph construction from vectors (k-NN graphs, mutual k-NN, etc.)
- Mini-dot language for simple graph specification
- Comprehensive graph format conversion using castgraph
- Support for NetworkX, pandas DataFrames, adjacency matrices, edge lists, and more

### Examples

```pycon
>>> from linked import mini_dot_to_graph_jdict
>>> mini_dot_to_graph_jdict('''
... 1 -> 2
... 2, 3 -> 5, 6, 7
... ''')
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
```

Graph conversion examples:

```pycon
>>> from linked import convert_graph
>>> import numpy as np
>>> # Convert edge list to nodes_and_links format
>>> edgelist = np.array([[0, 1], [1, 2]])
>>> graph = convert_graph(edgelist, 'nodes_and_links')
>>> 'nodes' in graph and 'links' in graph
True
```

```pycon
>>> # Convert mini-dot to edge list
>>> minidot = "1 -> 2\n2 -> 3"
>>> edgelist = convert_graph(minidot, 'edgelist', from_kind='minidot')
>>> len(edgelist) == 2
True
```

### Functions

| [`mini_dot_to_graph_jdict`](#linked.mini_dot_to_graph_jdict)(mini_dot, \*[, ...])       | Make a graph json dict from a mini-dot string.                      |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [`knn_graph`](#linked.knn_graph)(vectors, \*[, n_neighbors, metric, ...]) | Construct a k-nearest neighbors graph from vectors.                 |
| [`mutual_knn_graph`](#linked.mutual_knn_graph)(vectors, \*[, n_neighbors, ...])  | Construct a mutual k-nearest neighbors graph.                       |
| [`adaptive_knn_graph`](#linked.adaptive_knn_graph)(vectors, \*[, ...])             | Construct k-NN graph with adaptive distance scaling (UMAP-style).   |
| [`epsilon_graph`](#linked.epsilon_graph)(vectors, \*, radius[, metric, ...])  | Construct an epsilon-neighborhood graph.                            |
| [`random_graph`](#linked.random_graph)(n_nodes, \*[, model, p, m, k, ...])   | Generate a random graph according to classical random graph models. |
| [`convert_graph`](#linked.convert_graph)(obj, to_kind[, from_kind, context])  | Convert a graph from one representation to another.                 |
| [`graph_converter`](#linked.graph_converter)(from_kind, to_kind[, context])     | Get a converter function for repeated conversions.                  |
| [`graph_kinds`](#linked.graph_kinds)()                                      | List all registered graph kinds.                                    |
| [`reachable_from_kind`](#linked.reachable_from_kind)(kind)                          | Get all graph kinds reachable from a given kind.                    |
| [`sources_for_kind`](#linked.sources_for_kind)(kind)                             | Get all graph kinds that can be converted to a given kind.          |

### linked.adaptive_knn_graph(vectors, , n_neighbors=15, metric='euclidean', local_connectivity=1, bandwidth=1.0, approximate=True, n_jobs=-1)

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

### linked.convert_graph(obj, to_kind, from_kind=None, context=None)

Convert a graph from one representation to another.

* **Parameters:**
  * **obj** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The graph object to convert
  * **to_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The target graph kind
  * **from_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The source graph kind (auto-detected if None)
  * **context** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Additional context for the conversion (e.g., column names, field mappings)
* **Returns:**
  The converted graph object
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### Examples

```pycon
>>> minidot = "1 -> 2\n2 -> 3"
>>> graph = convert_graph(minidot, 'nodes_and_links')
>>> 'nodes' in graph and 'links' in graph
True
```

### linked.epsilon_graph(vectors, , radius, metric='euclidean', mode='distance', approximate=False, n_jobs=-1)

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

### linked.graph_converter(from_kind, to_kind, context=None)

Get a converter function for repeated conversions.

* **Parameters:**
  * **from_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Source graph kind
  * **to_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Destination graph kind
  * **context** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Context to bake into the converter
* **Returns:**
  A function that converts graphs from from_kind to to_kind
* **Return type:**
  `callable`

### Examples

```pycon
>>> to_edgelist = graph_converter('minidot', 'edgelist')
>>> edgelist = to_edgelist("1 -> 2")
```

### linked.graph_kinds()

List all registered graph kinds.

* **Returns:**
  Set of all registered graph kind identifiers
* **Return type:**
  [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)

### Examples

```pycon
>>> kinds = graph_kinds()
>>> 'nodes_and_links' in kinds
True
```

### linked.knn_graph(vectors, , n_neighbors=15, metric='euclidean', mode='distance', include_self=False, approximate=True, n_jobs=-1)

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

### linked.mini_dot_to_graph_jdict(mini_dot, , field_names={'id': 'id', 'links': 'links', 'nodes': 'nodes', 'source': 'source', 'target': 'target'})

Make a graph json dict from a mini-dot string.

```pycon
>>> mini_dot_to_graph_jdict('''
... 1 -> 2
... 2, 3 -> 5, 6, 7
... ''')
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
```

### linked.mutual_knn_graph(vectors, , n_neighbors=15, metric='euclidean', mode='distance', approximate=True, ensure_connectivity='mst', n_jobs=-1)

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

### linked.random_graph(n_nodes, , model='erdos_renyi', p=None, m=None, k=None, weighted=False, seed=None)

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

### linked.reachable_from_kind(kind)

Get all graph kinds reachable from a given kind.

* **Parameters:**
  **kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The source graph kind
* **Returns:**
  Set of reachable graph kind identifiers
* **Return type:**
  [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)

### Examples

```pycon
>>> reachable = reachable_from_kind('minidot')
>>> 'nodes_and_links' in reachable
True
```

### linked.sources_for_kind(kind)

Get all graph kinds that can be converted to a given kind.

* **Parameters:**
  **kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The target graph kind
* **Returns:**
  Set of source graph kind identifiers
* **Return type:**
  [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)

### Examples

```pycon
>>> sources = sources_for_kind('edgelist')
>>> len(sources) >= 0
True
```

### Modules

| [`cast`](linked.cast.html.md#module-linked.cast)       | Graph representation conversion system using castgraph.          |
|--------------------------------------------------------------------------------|------------------------------------------------------------------|
| [`datasrc`](linked.datasrc.html.md#module-linked.datasrc) | Utils to generate graph data from properties and mini-languages. |
| [`util`](linked.util.html.md#module-linked.util)       | Utils                                                            |
