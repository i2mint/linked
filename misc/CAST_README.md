## cast.py: Graph Conversion System

The `linked.cast` module provides a unified system for converting between different graph data representations. Built on top of `i2.castgraph`, it automatically discovers the shortest conversion path between any two graph formats.

### Supported Graph Formats

#### Core Formats
- **`nodes_and_links`**: JSON-style dict with `nodes` and `links` keys (D3.js compatible)
- **`edgelist`**: numpy array (n_edges, 2) with integer node indices
- **`weighted_edgelist`**: numpy array (n_edges, 3) with [source, target, weight]
- **`minidot`**: Simple text format (e.g., `"1 -> 2\n2, 3 -> 5"`)

#### Matrix Formats
- **`adjacency_matrix`**: Dense numpy array (n_nodes, n_nodes)
- **`sparse_adjacency`**: scipy sparse matrix (CSR, CSC, or COO)
- **`adjacency_list`**: Dict mapping node index to list of neighbors

#### External Libraries
- **`networkx_graph`**: NetworkX Graph (undirected)
- **`networkx_digraph`**: NetworkX DiGraph (directed)
- **`edges_dataframe`**: pandas DataFrame with source/target columns
- **`graph_dataframes`**: Dict with 'edges' and 'nodes' DataFrames

#### Special Formats
- **`vectors`**: numpy array (n_samples, n_features) - converts to graphs via k-NN

### Quick Start

```python
import numpy as np
from linked import convert_graph

# Convert edge list to nodes_and_links format
edgelist = np.array([[0, 1], [1, 2], [2, 0]])
graph = convert_graph(edgelist, 'nodes_and_links')
# graph = {'nodes': [{'id': '0'}, ...], 'links': [{'source': '0', 'target': '1'}, ...]}

# Convert mini-dot to NetworkX
minidot = "1 -> 2\n2 -> 3"
nx_graph = convert_graph(minidot, 'networkx_graph', from_kind='minidot')

# Convert vectors to graph using k-NN
vectors = np.random.rand(100, 50)
graph = convert_graph(
    vectors, 
    'weighted_edgelist',
    from_kind='vectors',
    context={'n_neighbors': 15, 'metric': 'euclidean'}
)
```

### Multi-Hop Conversions

The system automatically finds the shortest path between formats:

```python
# Mini-dot -> edgelist (goes through nodes_and_links automatically)
edgelist = convert_graph("1 -> 2", 'edgelist', from_kind='minidot')

# Edge list -> NetworkX DiGraph -> pandas DataFrame
import pandas as pd
edgelist = np.array([[0, 1], [1, 2]])
nx_digraph = convert_graph(edgelist, 'networkx_digraph')
df = convert_graph(nx_digraph, 'edges_dataframe', from_kind='networkx_digraph')
```

### Automatic Kind Detection

If you don't specify `from_kind`, the system will try to auto-detect:

```python
# Auto-detect edge list
edgelist = np.array([[0, 1], [1, 2]])
graph = convert_graph(edgelist, 'nodes_and_links')  # Detects as 'edgelist'

# Auto-detect nodes_and_links
graph = {'nodes': [{'id': '0'}], 'links': [{'source': '0', 'target': '1'}]}
edgelist = convert_graph(graph, 'edgelist')  # Detects as 'nodes_and_links'
```

### Using Context Parameters

Many conversions accept context parameters to customize behavior:

```python
# Custom field names for nodes_and_links
context = {
    'id_field': 'node_id',
    'source_field': 'from',
    'target_field': 'to',
    'weight_field': 'edge_weight'
}
graph = convert_graph(edgelist, 'nodes_and_links', context=context)

# Custom k-NN parameters for vectors
context = {
    'n_neighbors': 10,
    'metric': 'cosine',
    'graph_type': 'mutual_knn'  # Options: 'knn', 'mutual_knn', 'adaptive_knn', 'epsilon'
}
graph = convert_graph(vectors, 'edgelist', from_kind='vectors', context=context)

# Custom column names for DataFrames
context = {
    'source_col': 'src',
    'target_col': 'dst',
    'weight_col': 'w'
}
df = convert_graph(edgelist, 'edges_dataframe', context=context)
```

### Reusable Converters

Create a converter function for repeated conversions:

```python
from linked import get_graph_converter

# Create converter with baked-in context
to_nx = get_graph_converter(
    'edgelist', 
    'networkx_graph',
    context={'weight_attr': 'distance'}
)

# Use it multiple times
for edgelist in my_edgelists:
    nx_graph = to_nx(edgelist)
```

### Introspection

Discover available conversions:

```python
from linked import list_graph_kinds, reachable_from_kind, sources_for_kind

# List all registered graph kinds
kinds = list_graph_kinds()
print(kinds)
# {'edgelist', 'nodes_and_links', 'networkx_graph', 'adjacency_matrix', ...}

# Find what you can convert TO from a given format
reachable = reachable_from_kind('edgelist')
print(reachable)
# {'nodes_and_links', 'adjacency_matrix', 'networkx_graph', 'sparse_adjacency', ...}

# Find what you can convert FROM to a given format
sources = sources_for_kind('networkx_graph')
print(sources)
# {'edgelist', 'nodes_and_links', 'adjacency_matrix', ...}
```

### Adding Custom Converters

You can register your own graph formats and conversions:

```python
from linked.cast import register_kind, register_transformation

# Register a new kind
def is_my_format(obj):
    return isinstance(obj, MyGraphClass)

register_kind('my_format', isa=is_my_format)

# Register conversion to/from a hub format (e.g., edgelist)
@register_transformation('my_format', 'edgelist', cost=0.5)
def my_format_to_edgelist(obj, ctx):
    # Your conversion logic
    return edgelist

@register_transformation('edgelist', 'my_format', cost=0.5)
def edgelist_to_my_format(edgelist, ctx):
    # Your conversion logic
    return MyGraphClass(edgelist)
```

### Design Philosophy

1. **Hub-and-Spoke**: Most conversions go through hub formats (`edgelist`, `nodes_and_links`) to minimize pairwise converters
2. **Automatic Routing**: The system finds the shortest path (by cost) between any two formats
3. **Graceful Degradation**: Optional dependencies (NetworkX, pandas, scipy) are only required if you use those formats
4. **Context-Driven**: Conversion behavior can be customized via context dictionaries
5. **Testable**: Each converter has doctests and comprehensive test coverage

### Common Patterns

#### Working with NetworkX
```python
import networkx as nx
from linked import convert_graph

# Create from edge list
G = convert_graph(edgelist, 'networkx_graph')

# Add some NetworkX-specific attributes
G.nodes[0]['label'] = 'Start'
G.edges[0, 1]['weight'] = 1.5

# Convert to other formats (preserves attributes)
graph_dict = convert_graph(G, 'nodes_and_links', from_kind='networkx_graph')
```

#### Working with pandas DataFrames
```python
import pandas as pd
from linked import convert_graph

# From DataFrame
edges_df = pd.DataFrame({
    'source': [0, 1, 2],
    'target': [1, 2, 0],
    'weight': [0.5, 0.8, 0.3]
})
graph = convert_graph(edges_df, 'nodes_and_links')

# To DataFrame (with nodes metadata)
nodes = [{'id': 0, 'label': 'A'}, {'id': 1, 'label': 'B'}]
links = [{'source': 0, 'target': 1, 'weight': 0.5}]
graph_dfs = convert_graph(
    {'nodes': nodes, 'links': links}, 
    'graph_dataframes'
)
edges_df = graph_dfs['edges']
nodes_df = graph_dfs['nodes']
```

#### Working with Vectors
```python
from linked import convert_graph

# Build k-NN graph from high-dimensional data
vectors = np.random.rand(1000, 128)
knn_graph = convert_graph(
    vectors,
    'weighted_edgelist',
    from_kind='vectors',
    context={'n_neighbors': 15, 'metric': 'cosine'}
)

# Convert to NetworkX for analysis
G = convert_graph(knn_graph, 'networkx_graph')
import networkx as nx
communities = nx.community.louvain_communities(G)
```

### Performance Tips

1. **Use sparse formats** for large graphs (>10K nodes)
2. **Cache converters** with `get_graph_converter()` for repeated conversions
3. **Use approximate k-NN** for large vector datasets (set `approximate=True` in context)
4. **Specify `from_kind`** explicitly to avoid auto-detection overhead
5. **Choose the right hub**: Convert to `edgelist` for algorithms, `nodes_and_links` for visualization

### Troubleshooting

#### Missing Dependencies
```python
# If you see "X kind not registered", install the required package:
# pip install networkx  # for networkx_graph, networkx_digraph
# pip install pandas    # for edges_dataframe, graph_dataframes
# pip install scipy     # for sparse_adjacency
```

#### Conversion Errors
```python
from linked.cast import ConversionError

try:
    result = convert_graph(obj, 'target_kind')
except ConversionError as e:
    print(f"No conversion path found: {e}")
    # Check what conversions are available
    from linked import reachable_from_kind
    print(reachable_from_kind('source_kind'))
```

### Architecture

The conversion system is built using these patterns:
- **Type Converter/Conversion Service**: Central registry for transformations
- **Adapter Pattern**: Each edge adapts one representation to another
- **Strategy Pattern**: Route selection via cost-based shortest path
- **Hub-and-Spoke**: Reduce O(n²) converters to O(n) via hub formats

For more details, see the `i2.castgraph` documentation.
