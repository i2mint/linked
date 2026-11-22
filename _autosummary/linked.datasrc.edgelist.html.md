# linked.datasrc.edgelist

Edge list graph representations and conversions.

An edge list is one of the most compact graph representations, consisting of
an array of edges where each edge is [source, target] or [source, target, weight].

## Kinds defined here

- ‘edgelist’: numpy array (n_edges, 2) with integer node indices
- ‘weighted_edgelist’: numpy array (n_edges, 3) with [source, target, weight]

## Conversions

- edgelist <-> nodes_and_links
- weighted_edgelist <-> nodes_and_links
- edgelist <-> weighted_edgelist (add/remove weights)

### Functions

| [`edgelist_to_nodes_and_links`](#linked.datasrc.edgelist.edgelist_to_nodes_and_links)(edgelist[, ctx])     | Convert edge list to nodes_and_links dict.      |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------|
| [`edgelist_to_weighted_edgelist`](#linked.datasrc.edgelist.edgelist_to_weighted_edgelist)(edgelist[, ctx])   | Add default weights to an unweighted edge list. |
| [`nodes_and_links_to_edgelist`](#linked.datasrc.edgelist.nodes_and_links_to_edgelist)(nodes_and_links)     | Convert nodes_and_links dict to edge list.      |
| [`weighted_edgelist_to_edgelist`](#linked.datasrc.edgelist.weighted_edgelist_to_edgelist)(weighted_edgelist) | Remove weights from a weighted edge list.       |

### linked.datasrc.edgelist.edgelist_to_nodes_and_links(edgelist, ctx=None)

Convert edge list to nodes_and_links dict.

* **Parameters:**
  * **edgelist** (`ndarray`) – Array of shape (n_edges, 2) or (n_edges, 3)
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’id_field’: field name for node id (default: ‘id’)
    - ’source_field’: field name for edge source (default: ‘source’)
    - ’target_field’: field name for edge target (default: ‘target’)
    - ’weight_field’: field name for edge weight (default: ‘weight’)
    - ’node_id_map’: dict mapping indices to node ids
* **Returns:**
  Dict with ‘nodes’ and ‘links’ keys
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Examples

```pycon
>>> edgelist = np.array([[0, 1], [1, 2], [2, 0]])
>>> result = edgelist_to_nodes_and_links(edgelist)
>>> len(result['nodes'])
3
>>> len(result['links'])
3
```

### linked.datasrc.edgelist.edgelist_to_weighted_edgelist(edgelist, ctx=None)

Add default weights to an unweighted edge list.

* **Parameters:**
  * **edgelist** (`ndarray`) – Array of shape (n_edges, 2)
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Context with ‘default_weight’ key (default: 1.0)
* **Returns:**
  Array of shape (n_edges, 3)
* **Return type:**
  `ndarray`

### Examples

```pycon
>>> edgelist = np.array([[0, 1], [1, 2]])
>>> weighted = edgelist_to_weighted_edgelist(edgelist)
>>> weighted.shape
(2, 3)
>>> float(weighted[0, 2])
1.0
```

### linked.datasrc.edgelist.nodes_and_links_to_edgelist(nodes_and_links, ctx=None)

Convert nodes_and_links dict to edge list.

* **Parameters:**
  * **nodes_and_links** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dict with ‘nodes’ and ‘links’ keys
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’id_field’: field name for node id (default: ‘id’)
    - ’source_field’: field name for edge source (default: ‘source’)
    - ’target_field’: field name for edge target (default: ‘target’)
    - ’weight_field’: field name for edge weight (default: ‘weight’)
    - ’include_weights’: whether to include weights if present (default: False)
* **Returns:**
  Edge list array of shape (n_edges, 2) or (n_edges, 3)
* **Return type:**
  `ndarray`

### Examples

```pycon
>>> graph = {'nodes': [{'id': '0'}, {'id': '1'}], 'links': [{'source': '0', 'target': '1'}]}
>>> edgelist = nodes_and_links_to_edgelist(graph)
>>> edgelist.shape
(1, 2)
```

### linked.datasrc.edgelist.weighted_edgelist_to_edgelist(weighted_edgelist, ctx=None)

Remove weights from a weighted edge list.

* **Parameters:**
  * **weighted_edgelist** (`ndarray`) – Array of shape (n_edges, 3)
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Not used
* **Returns:**
  Array of shape (n_edges, 2)
* **Return type:**
  `ndarray`

### Examples

```pycon
>>> weighted = np.array([[0, 1, 0.5], [1, 2, 0.8]])
>>> edgelist = weighted_edgelist_to_edgelist(weighted)
>>> edgelist.shape
(2, 2)
```
