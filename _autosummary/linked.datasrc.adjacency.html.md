# linked.datasrc.adjacency

Adjacency matrix and adjacency list graph representations.

Adjacency formats are classical graph representations used in algorithms and analysis.

## Kinds defined here

- ‘adjacency_matrix’: Dense numpy array (n_nodes, n_nodes)
- ‘sparse_adjacency’: scipy sparse matrix (CSR, CSC, COO, etc.)
- ‘adjacency_list’: Dict mapping node index to list of neighbor indices

## Conversions

- adjacency matrix <-> edgelist
- sparse adjacency <-> edgelist
- adjacency list <-> edgelist
- adjacency matrix <-> sparse adjacency

### Functions

| [`adjacency_list_to_edgelist`](#linked.datasrc.adjacency.adjacency_list_to_edgelist)(adj_list[, ctx])      | Convert adjacency list to edge list.             |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [`adjacency_matrix_to_edgelist`](#linked.datasrc.adjacency.adjacency_matrix_to_edgelist)(adj_matrix[, ctx])  | Convert adjacency matrix to edge list.           |
| [`adjacency_matrix_to_sparse_adjacency`](#linked.datasrc.adjacency.adjacency_matrix_to_sparse_adjacency)(adj_matrix) | Convert dense adjacency matrix to sparse format. |
| [`edgelist_to_adjacency_list`](#linked.datasrc.adjacency.edgelist_to_adjacency_list)(edgelist[, ctx])      | Convert edge list to adjacency list.             |
| [`edgelist_to_adjacency_matrix`](#linked.datasrc.adjacency.edgelist_to_adjacency_matrix)(edgelist[, ctx])    | Convert edge list to adjacency matrix.           |
| [`edgelist_to_sparse_adjacency`](#linked.datasrc.adjacency.edgelist_to_sparse_adjacency)(edgelist[, ctx])    | Convert edge list to sparse adjacency matrix.    |
| [`sparse_adjacency_to_adjacency_matrix`](#linked.datasrc.adjacency.sparse_adjacency_to_adjacency_matrix)(sparse_adj) | Convert sparse adjacency matrix to dense format. |
| [`sparse_adjacency_to_edgelist`](#linked.datasrc.adjacency.sparse_adjacency_to_edgelist)(sparse_adj[, ctx])  | Convert sparse adjacency matrix to edge list.    |

### linked.datasrc.adjacency.adjacency_list_to_edgelist(adj_list, ctx=None)

Convert adjacency list to edge list.

* **Parameters:**
  * **adj_list** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dict mapping node index to list of neighbor indices
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’include_weights’: whether edges have weights (default: False)
    - ’directed’: whether graph is directed (default: True for adj lists)
* **Returns:**
  Edge list array
* **Return type:**
  `ndarray`

### Examples

```pycon
>>> adj_list = {0: [1, 2], 1: [2], 2: []}
>>> edgelist = adjacency_list_to_edgelist(adj_list)
>>> len(edgelist) == 3
True
```

### linked.datasrc.adjacency.adjacency_matrix_to_edgelist(adj_matrix, ctx=None)

Convert adjacency matrix to edge list.

* **Parameters:**
  * **adj_matrix** (`ndarray`) – Square adjacency matrix of shape (n_nodes, n_nodes)
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’directed’: whether graph is directed (default: auto-detect from symmetry)
    - ’include_weights’: whether to include weights (default: True if non-binary)
    - ’threshold’: minimum weight to include edge (default: 0)
* **Returns:**
  Edge list array
* **Return type:**
  `ndarray`

### Examples

```pycon
>>> adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
>>> edgelist = adjacency_matrix_to_edgelist(adj)
>>> len(edgelist) > 0
True
```

### linked.datasrc.adjacency.adjacency_matrix_to_sparse_adjacency(adj_matrix, ctx=None)

Convert dense adjacency matrix to sparse format.

### linked.datasrc.adjacency.edgelist_to_adjacency_list(edgelist, ctx=None)

Convert edge list to adjacency list.

* **Parameters:**
  * **edgelist** (`ndarray`) – Edge list array
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’directed’: whether graph is directed (default: False)
    - ’include_weights’: store weights in adjacency list (default: auto-detect)
* **Returns:**
  Adjacency list dict
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Examples

```pycon
>>> edgelist = np.array([[0, 1], [1, 2], [0, 2]])
>>> adj_list = edgelist_to_adjacency_list(edgelist)
>>> len(adj_list[0]) == 2
True
```

### linked.datasrc.adjacency.edgelist_to_adjacency_matrix(edgelist, ctx=None)

Convert edge list to adjacency matrix.

* **Parameters:**
  * **edgelist** (`ndarray`) – Edge list array
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’n_nodes’: number of nodes (default: max index + 1)
    - ’directed’: whether graph is directed (default: False)
    - ’dtype’: data type for matrix (default: float if weighted, int otherwise)
* **Returns:**
  Adjacency matrix
* **Return type:**
  `ndarray`

### Examples

```pycon
>>> edgelist = np.array([[0, 1], [1, 2]])
>>> adj = edgelist_to_adjacency_matrix(edgelist)
>>> adj.shape[0] == adj.shape[1]
True
```

### linked.datasrc.adjacency.edgelist_to_sparse_adjacency(edgelist, ctx=None)

Convert edge list to sparse adjacency matrix.

* **Parameters:**
  * **edgelist** (`ndarray`) – Edge list array
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’n_nodes’: number of nodes (default: max index + 1)
    - ’directed’: whether graph is directed (default: False)
    - ’format’: sparse matrix format - ‘csr’, ‘csc’, ‘coo’ (default: ‘csr’)
* **Returns:**
  Sparse adjacency matrix
* **Return type:**
  scipy.sparse matrix

### Examples

```pycon
>>> edgelist = np.array([[0, 1, 0.5], [1, 2, 0.8]])
>>> sparse_adj = edgelist_to_sparse_adjacency(edgelist)
>>> sparse_adj.shape[0] == sparse_adj.shape[1]
True
```

### linked.datasrc.adjacency.sparse_adjacency_to_adjacency_matrix(sparse_adj, ctx=None)

Convert sparse adjacency matrix to dense format.

* **Return type:**
  `ndarray`

### linked.datasrc.adjacency.sparse_adjacency_to_edgelist(sparse_adj, ctx=None)

Convert sparse adjacency matrix to edge list.

* **Parameters:**
  * **sparse_adj** (*scipy.sparse matrix*) – Sparse adjacency matrix
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Context (same as adjacency_matrix_to_edgelist)
* **Returns:**
  Edge list array
* **Return type:**
  `ndarray`

### Examples

```pycon
>>> from scipy import sparse
>>> adj = sparse.csr_matrix(([1, 1], ([0, 1], [1, 2])), shape=(3, 3))
>>> edgelist = sparse_adjacency_to_edgelist(adj)
>>> len(edgelist) > 0
True
```
