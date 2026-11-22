# linked.cast

Graph representation conversion system using castgraph.

This module provides a centralized TransformationGraph for converting between
various graph data representations. Each datasrc module registers its transformations
here, creating a unified conversion service.

## Core graph kinds

- ‘nodes_and_links’: Dict with ‘nodes’ and ‘links’ keys (standard JSON graph format)
- ‘edgelist’: numpy array of shape (n_edges, 2) with [source, target] indices
- ‘weighted_edgelist’: numpy array of shape (n_edges, 3) with [source, target, weight]
- ‘minidot’: String in mini-dot format (e.g., “1 -> 2n2, 3 -> 5”)
- ‘adjacency_matrix’: Dense numpy array of shape (n_nodes, n_nodes)
- ‘sparse_adjacency’: scipy sparse matrix
- ‘adjacency_list’: Dict mapping node indices to lists of neighbors
- ‘networkx_graph’: NetworkX Graph object
- ‘networkx_digraph’: NetworkX DiGraph object
- ‘edges_dataframe’: pandas DataFrame with source/target columns
- ‘nodes_dataframe’: pandas DataFrame with node metadata
- ‘graph_dataframes’: Dict with ‘edges’ and ‘nodes’ DataFrames

## Design

Each datasrc module should:

1. Define kind predicates (isa functions) for its graph representations
2. Register transformations to/from at least one “hub” format (typically ‘nodes_and_links’ or ‘edgelist’)
3. Use contextlib.suppress to gracefully handle missing dependencies

## Example usage

```pycon
>>> from linked.cast import graph_transformer
>>> # Convert mini-dot string to NetworkX graph
>>> minidot = "1 -> 2\n2, 3 -> 5, 6"
>>> nx_graph = graph_transformer.transform_any(minidot, 'networkx_graph')
```

```pycon
>>> # Get a transformer function for repeated conversions
>>> to_edgelist = graph_transformer.get_transformer('minidot', 'edgelist')
>>> edgelist = to_edgelist(minidot)
```

### Functions

| [`convert_graph`](#linked.cast.convert_graph)(obj, to_kind[, from_kind, context])   | Convert a graph from one representation to another.         |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [`graph_converter`](#linked.cast.graph_converter)(from_kind, to_kind[, context])      | Get a converter function for repeated conversions.          |
| [`graph_kinds`](#linked.cast.graph_kinds)()                                       | List all registered graph kinds.                            |
| [`reachable_from_kind`](#linked.cast.reachable_from_kind)(kind)                           | Get all graph kinds reachable from a given kind.            |
| [`sources_for_kind`](#linked.cast.sources_for_kind)(kind)                              | Get all graph kinds that can be converted to a given kind.  |
| [`register_kind`](#linked.cast.register_kind)(kind[, isa])                          | Register a new graph kind with optional predicate.          |
| [`register_transformation`](#linked.cast.register_transformation)(src_kind, dst_kind)         | Decorator to register a transformation between graph kinds. |

### linked.cast.convert_graph(obj, to_kind, from_kind=None, context=None)

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

### linked.cast.graph_converter(from_kind, to_kind, context=None)

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

### linked.cast.graph_kinds()

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

### linked.cast.reachable_from_kind(kind)

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

### linked.cast.register_kind(kind, isa=None)

Register a new graph kind with optional predicate.

* **Parameters:**
  * **kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The kind identifier
  * **isa** (`callable`) – Predicate function to detect if an object is of this kind

### Examples

```pycon
>>> with suppress(Exception):
...     register_kind('my_format', lambda x: isinstance(x, MyFormat))
```

### linked.cast.register_transformation(src_kind, dst_kind, cost=1.0)

Decorator to register a transformation between graph kinds.

* **Parameters:**
  * **src_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Source graph kind
  * **dst_kind** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Destination graph kind
  * **cost** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Cost of this transformation (lower = preferred)

### Examples

```pycon
>>> @register_transformation('my_format', 'edgelist')
... def my_format_to_edgelist(obj, ctx):
...     return convert_to_edgelist(obj)
```

### linked.cast.sources_for_kind(kind)

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
