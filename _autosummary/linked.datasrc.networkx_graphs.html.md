# linked.datasrc.networkx_graphs

NetworkX graph representations and conversions.

NetworkX is one of the most popular Python graph libraries, supporting both
undirected (Graph) and directed (DiGraph) graphs with extensive algorithms.

## Kinds defined here

- ‘networkx_graph’: networkx.Graph (undirected)
- ‘networkx_digraph’: networkx.DiGraph (directed)

## Conversions

- networkx graphs <-> edgelist
- networkx graphs <-> nodes_and_links

### Functions

| [`edgelist_to_networkx_digraph`](#linked.datasrc.networkx_graphs.edgelist_to_networkx_digraph)(edgelist[, ctx])   | Convert edge list to NetworkX directed graph.              |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`edgelist_to_networkx_graph`](#linked.datasrc.networkx_graphs.edgelist_to_networkx_graph)(edgelist[, ctx])     | Convert edge list to NetworkX undirected graph.            |
| [`networkx_to_edgelist`](#linked.datasrc.networkx_graphs.networkx_to_edgelist)(graph[, ctx])              | Convert NetworkX graph to edge list.                       |
| [`networkx_to_nodes_and_links`](#linked.datasrc.networkx_graphs.networkx_to_nodes_and_links)(graph[, ctx])       | Convert NetworkX graph to nodes_and_links dict.            |
| [`nodes_and_links_to_networkx_digraph`](#linked.datasrc.networkx_graphs.nodes_and_links_to_networkx_digraph)(...[, ctx]) | Convert nodes_and_links dict to NetworkX directed graph.   |
| [`nodes_and_links_to_networkx_graph`](#linked.datasrc.networkx_graphs.nodes_and_links_to_networkx_graph)(...[, ctx])   | Convert nodes_and_links dict to NetworkX undirected graph. |

### linked.datasrc.networkx_graphs.edgelist_to_networkx_digraph(edgelist, ctx=None)

Convert edge list to NetworkX directed graph.

* **Parameters:**
  * **edgelist** (`ndarray`) – Edge list array
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Context (same as edgelist_to_networkx_graph)
* **Returns:**
  NetworkX directed graph
* **Return type:**
  `DiGraph`

### Examples

```pycon
>>> edgelist = np.array([[0, 1], [1, 2]])
>>> G = edgelist_to_networkx_digraph(edgelist)
>>> G.is_directed()
True
```

### linked.datasrc.networkx_graphs.edgelist_to_networkx_graph(edgelist, ctx=None)

Convert edge list to NetworkX undirected graph.

* **Parameters:**
  * **edgelist** (`ndarray`) – Edge list array of shape (n_edges, 2) or (n_edges, 3)
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’weight_attr’: attribute name for edge weight (default: ‘weight’)
    - ’node_id_map’: dict mapping indices to node ids
* **Returns:**
  NetworkX undirected graph
* **Return type:**
  `Graph`

### Examples

```pycon
>>> edgelist = np.array([[0, 1], [1, 2]])
>>> G = edgelist_to_networkx_graph(edgelist)
>>> G.number_of_nodes()
3
```

### linked.datasrc.networkx_graphs.networkx_to_edgelist(graph, ctx=None)

Convert NetworkX graph to edge list.

* **Parameters:**
  * **graph** (`Graph`) – NetworkX graph object
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’weight_attr’: attribute name for edge weight (default: ‘weight’)
    - ’include_weights’: whether to include weights (default: auto-detect)
* **Returns:**
  Edge list array
* **Return type:**
  `ndarray`

### Examples

```pycon
>>> G = nx.Graph()
>>> G.add_edges_from([(0, 1), (1, 2)])
>>> edgelist = networkx_to_edgelist(G)
>>> edgelist.shape[0] == 2
True
```

### linked.datasrc.networkx_graphs.networkx_to_nodes_and_links(graph, ctx=None)

Convert NetworkX graph to nodes_and_links dict.

* **Parameters:**
  * **graph** (`Graph`) – NetworkX graph
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’id_field’: field name for node id (default: ‘id’)
    - ’source_field’: field name for edge source (default: ‘source’)
    - ’target_field’: field name for edge target (default: ‘target’)
    - ’weight_field’: field name for edge weight (default: ‘weight’)
    - ’weight_attr’: NetworkX attribute for weight (default: ‘weight’)
    - ’include_node_attrs’: whether to include node attributes (default: True)
    - ’include_edge_attrs’: whether to include edge attributes (default: True)
* **Returns:**
  Dict with ‘nodes’ and ‘links’ keys
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Examples

```pycon
>>> G = nx.Graph()
>>> G.add_edges_from([(0, 1), (1, 2)])
>>> result = networkx_to_nodes_and_links(G)
>>> len(result['nodes']) == 3
True
```

### linked.datasrc.networkx_graphs.nodes_and_links_to_networkx_digraph(nodes_and_links, ctx=None)

Convert nodes_and_links dict to NetworkX directed graph.

Same as nodes_and_links_to_networkx_graph but returns DiGraph.

### Examples

```pycon
>>> data = {'nodes': [{'id': '0'}, {'id': '1'}], 'links': [{'source': '0', 'target': '1'}]}
>>> G = nodes_and_links_to_networkx_digraph(data)
>>> G.is_directed()
True
```

* **Return type:**
  `DiGraph`

### linked.datasrc.networkx_graphs.nodes_and_links_to_networkx_graph(nodes_and_links, ctx=None)

Convert nodes_and_links dict to NetworkX undirected graph.

* **Parameters:**
  * **nodes_and_links** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dict with ‘nodes’ and ‘links’ keys
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Context (see networkx_to_nodes_and_links)
* **Returns:**
  NetworkX undirected graph
* **Return type:**
  `Graph`

### Examples

```pycon
>>> data = {'nodes': [{'id': '0'}, {'id': '1'}], 'links': [{'source': '0', 'target': '1'}]}
>>> G = nodes_and_links_to_networkx_graph(data)
>>> G.number_of_edges()
1
```
