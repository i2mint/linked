# linked.datasrc.dataframes

Pandas DataFrame graph representations and conversions.

DataFrames are common in data science workflows, with edges as rows in one DataFrame
and optionally nodes as rows in another DataFrame with metadata.

## Kinds defined here

- ‘edges_dataframe’: pandas DataFrame with source/target columns
- ‘graph_dataframes’: Dict with ‘edges’ and optionally ‘nodes’ DataFrames

## Conversions

- edges DataFrame <-> edgelist
- edges DataFrame <-> nodes_and_links
- graph DataFrames <-> nodes_and_links

### Functions

| [`edgelist_to_edges_dataframe`](#linked.datasrc.dataframes.edgelist_to_edges_dataframe)(edgelist[, ctx])    | Convert edge list to edges DataFrame.                                    |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`edges_dataframe_to_edgelist`](#linked.datasrc.dataframes.edges_dataframe_to_edgelist)(df[, ctx])          | Convert edges DataFrame to edge list.                                    |
| [`edges_dataframe_to_nodes_and_links`](#linked.datasrc.dataframes.edges_dataframe_to_nodes_and_links)(df[, ctx])   | Convert edges DataFrame to nodes_and_links dict.                         |
| [`graph_dataframes_to_nodes_and_links`](#linked.datasrc.dataframes.graph_dataframes_to_nodes_and_links)(graph_dfs)  | Convert graph DataFrames (with edges and nodes) to nodes_and_links dict. |
| [`nodes_and_links_to_edges_dataframe`](#linked.datasrc.dataframes.nodes_and_links_to_edges_dataframe)(...[, ctx])  | Convert nodes_and_links dict to edges DataFrame.                         |
| [`nodes_and_links_to_graph_dataframes`](#linked.datasrc.dataframes.nodes_and_links_to_graph_dataframes)(...[, ctx]) | Convert nodes_and_links dict to graph DataFrames.                        |

### linked.datasrc.dataframes.edgelist_to_edges_dataframe(edgelist, ctx=None)

Convert edge list to edges DataFrame.

* **Parameters:**
  * **edgelist** (`ndarray`) – Edge list array
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’source_col’: column name for source (default: ‘source’)
    - ’target_col’: column name for target (default: ‘target’)
    - ’weight_col’: column name for weight (default: ‘weight’)
    - ’node_id_map’: dict mapping indices to node ids
* **Returns:**
  Edges DataFrame
* **Return type:**
  `DataFrame`

### Examples

```pycon
>>> edgelist = np.array([[0, 1], [1, 2]])
>>> df = edgelist_to_edges_dataframe(edgelist)
>>> 'source' in df.columns and 'target' in df.columns
True
```

### linked.datasrc.dataframes.edges_dataframe_to_edgelist(df, ctx=None)

Convert edges DataFrame to edge list.

* **Parameters:**
  * **df** (`DataFrame`) – DataFrame with edge data
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’source_col’: column name for source (default: auto-detect)
    - ’target_col’: column name for target (default: auto-detect)
    - ’weight_col’: column name for weight (default: ‘weight’)
    - ’include_weights’: whether to include weights (default: auto-detect)
    - ’node_id_map’: dict mapping node ids to indices
* **Returns:**
  Edge list array
* **Return type:**
  `ndarray`

### Examples

```pycon
>>> df = pd.DataFrame({'source': [0, 1], 'target': [1, 2]})
>>> edgelist = edges_dataframe_to_edgelist(df)
>>> edgelist.shape
(2, 2)
```

### linked.datasrc.dataframes.edges_dataframe_to_nodes_and_links(df, ctx=None)

Convert edges DataFrame to nodes_and_links dict.

* **Parameters:**
  * **df** (`DataFrame`) – Edges DataFrame
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Context (same as edges_dataframe_to_edgelist)
* **Returns:**
  Dict with ‘nodes’ and ‘links’ keys
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Examples

```pycon
>>> df = pd.DataFrame({'source': ['a', 'b'], 'target': ['b', 'c']})
>>> result = edges_dataframe_to_nodes_and_links(df)
>>> len(result['nodes']) == 3
True
```

### linked.datasrc.dataframes.graph_dataframes_to_nodes_and_links(graph_dfs, ctx=None)

Convert graph DataFrames (with edges and nodes) to nodes_and_links dict.

* **Parameters:**
  * **graph_dfs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dict with ‘edges’ DataFrame and optionally ‘nodes’ DataFrame
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Context (see edges_dataframe_to_nodes_and_links)
* **Returns:**
  Dict with ‘nodes’ and ‘links’ keys
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Examples

```pycon
>>> edges_df = pd.DataFrame({'source': [0, 1], 'target': [1, 2]})
>>> nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['A', 'B', 'C']})
>>> graph_dfs = {'edges': edges_df, 'nodes': nodes_df}
>>> result = graph_dataframes_to_nodes_and_links(graph_dfs)
>>> len(result['nodes'])
3
```

### linked.datasrc.dataframes.nodes_and_links_to_edges_dataframe(nodes_and_links, ctx=None)

Convert nodes_and_links dict to edges DataFrame.

* **Parameters:**
  * **nodes_and_links** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dict with ‘nodes’ and ‘links’ keys
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – 

    Context with optional keys:
    - ’source_col’: column name for source (default: ‘source’)
    - ’target_col’: column name for target (default: ‘target’)
    - ’source_field’: field name in links for source (default: ‘source’)
    - ’target_field’: field name in links for target (default: ‘target’)
* **Returns:**
  Edges DataFrame
* **Return type:**
  `DataFrame`

### Examples

```pycon
>>> data = {'nodes': [{'id': 'a'}, {'id': 'b'}], 'links': [{'source': 'a', 'target': 'b'}]}
>>> df = nodes_and_links_to_edges_dataframe(data)
>>> len(df)
1
```

### linked.datasrc.dataframes.nodes_and_links_to_graph_dataframes(nodes_and_links, ctx=None)

Convert nodes_and_links dict to graph DataFrames.

* **Parameters:**
  * **nodes_and_links** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dict with ‘nodes’ and ‘links’ keys
  * **ctx** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Context (see nodes_and_links_to_edges_dataframe)
* **Returns:**
  Dict with ‘edges’ and ‘nodes’ DataFrames
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Examples

```pycon
>>> data = {'nodes': [{'id': 0}, {'id': 1}], 'links': [{'source': 0, 'target': 1}]}
>>> graph_dfs = nodes_and_links_to_graph_dataframes(data)
>>> 'edges' in graph_dfs and 'nodes' in graph_dfs
True
```
