## Graph Conversion System Implementation Summary

### Overview

A comprehensive graph data conversion system has been implemented for the `linked` package using the `i2.castgraph` framework. This system enables seamless conversion between 12+ different graph representations with automatic multi-hop routing.

### Architecture

#### Core Components

1. **`linked/cast.py`** - Central transformation registry
   - Sets up the global `TransformationGraph`
   - Defines core kind predicates
   - Provides high-level API (convert_graph, get_graph_converter, etc.)
   - Loads and registers all datasrc modules

2. **datasrc Modules** - Individual graph format handlers
   - `edgelist.py` - Edge list formats (2-col, 3-col weighted)
   - `networkx_graphs.py` - NetworkX Graph/DiGraph support
   - `dataframes.py` - pandas DataFrame conversions
   - `adjacency.py` - Adjacency matrix/list/sparse formats
   - `minidot.py` - Mini-dot language (updated)
   - `vectors.py` - Vector → graph via k-NN (updated)

3. **Tests** - Comprehensive test coverage
   - `tests/test_cast.py` - 14 test functions covering all conversion paths
   - Tests for multi-hop conversions, round-trips, auto-detection, context parameters

### Supported Graph Kinds

#### Core Formats
- **nodes_and_links**: JSON-style dict with 'nodes' and 'links' keys
- **edgelist**: numpy array (n_edges, 2) with integer indices
- **weighted_edgelist**: numpy array (n_edges, 3) with weights
- **minidot**: Simple text format (e.g., "1 -> 2\n2, 3 -> 5")

#### Matrix Formats
- **adjacency_matrix**: Dense numpy array (n_nodes, n_nodes)
- **sparse_adjacency**: scipy sparse matrix (CSR/CSC/COO)
- **adjacency_list**: Dict mapping node to list of neighbors

#### External Library Formats
- **networkx_graph**: NetworkX Graph (undirected)
- **networkx_digraph**: NetworkX DiGraph (directed)
- **edges_dataframe**: pandas DataFrame with source/target columns
- **graph_dataframes**: Dict with 'edges' and 'nodes' DataFrames

#### Special Formats
- **vectors**: numpy array (n_samples, n_features) → graph via k-NN algorithms

### Key Features

#### 1. Automatic Multi-Hop Routing
```python
# Automatically routes through intermediate formats
minidot_str = "1 -> 2\n2 -> 3"
adj_matrix = convert_graph(minidot_str, 'adjacency_matrix', from_kind='minidot')
# Routes: minidot → nodes_and_links → edgelist → adjacency_matrix
```

#### 2. Auto-Detection
```python
# Automatically detects source format
edgelist = np.array([[0, 1], [1, 2]])
graph = convert_graph(edgelist, 'nodes_and_links')  # No from_kind needed
```

#### 3. Context Parameters
```python
# Customize conversions with context
context = {
    'n_neighbors': 10,
    'metric': 'cosine',
    'graph_type': 'mutual_knn'
}
graph = convert_graph(vectors, 'edgelist', from_kind='vectors', context=context)
```

#### 4. Reusable Converters
```python
# Create converter function for repeated use
to_nx = get_graph_converter('edgelist', 'networkx_graph')
for edges in my_edgelists:
    nx_graph = to_nx(edges)
```

#### 5. Introspection
```python
# Discover available conversions
kinds = list_graph_kinds()
reachable = reachable_from_kind('edgelist')
sources = sources_for_kind('networkx_graph')
```

#### 6. Graceful Dependency Handling
All optional dependencies (NetworkX, pandas, scipy) are handled gracefully with `contextlib.suppress`, so formats only register if their dependencies are available.

### Design Patterns

The implementation follows these well-established patterns:

1. **Type Converter / Conversion Service**: Central registry for transformations
2. **Adapter Pattern**: Each converter adapts one format to another
3. **Strategy Pattern**: Route selection via cost-based shortest path
4. **Hub-and-Spoke**: Most conversions go through hub formats (edgelist, nodes_and_links)
5. **Dependency Injection**: Context parameters allow customization without hardcoding

### Conversion Graph Structure

```
                    ┌─────────────┐
                    │   vectors   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────────┐
                    │ weighted_edgelist   │
                    └──────┬──────────────┘
                           │
    ┌──────────────┬───────┼────────┬──────────────┐
    │              │       │        │              │
┌───▼───┐    ┌────▼────┐  │  ┌────▼─────┐   ┌────▼─────┐
│minidot│◄──►│nodes_and│◄─┴─►│ edgelist │◄─►│adjacency_│
└───────┘    │ _links  │     │          │   │  matrix  │
             └────┬────┘     └────┬─────┘   └────┬─────┘
                  │               │              │
        ┌─────────┼───────────────┼──────────────┤
        │         │               │              │
   ┌────▼──┐  ┌──▼───┐      ┌───▼───┐     ┌───▼────┐
   │ nx    │  │edges │      │adj    │     │sparse  │
   │graph  │  │df    │      │list   │     │adj     │
   └───────┘  └──────┘      └───────┘     └────────┘
```

### Test Results

All tests pass successfully:

```
14 passed in 10.51s
- test_imports
- test_core_kinds
- test_minidot_conversions
- test_edgelist_conversions
- test_adjacency_conversions
- test_multi_hop_conversion
- test_networkx_conversions
- test_pandas_conversions
- test_vectors_conversions
- test_round_trip_conversions
- test_context_parameters
- test_auto_detection
- test_sparse_adjacency
- test_reachability
```

### Documentation

1. **README.md section** - Comprehensive user guide
   - Quick start examples
   - Format descriptions
   - Context parameters reference
   - Common patterns and best practices
   - Troubleshooting guide

2. **misc/cast_demo.py** - Interactive demonstration
   - Shows all 12 graph kinds
   - Demonstrates conversions
   - Tests multi-hop routing
   - Shows reachability queries

3. **Updated README.md** - Main package documentation
   - Added cast system overview
   - Quick start examples
   - Link to detailed documentation

4. **Inline Documentation**
   - All functions have docstrings with examples
   - Module-level documentation explains purpose
   - Context parameters documented in each converter

### API Exports

The following are now available from `linked` package:

```python
from linked import (
    # Conversion functions
    convert_graph,
    get_graph_converter,
    
    # Introspection
    list_graph_kinds,
    reachable_from_kind,
    sources_for_kind,
    
    # Advanced (if needed)
    graph_transformer,
)
```

### Extensibility

Users can easily add their own graph formats:

```python
from linked.cast import register_kind, register_transformation

# Register new kind
register_kind('my_format', isa=lambda x: isinstance(x, MyFormat))

# Register conversions
@register_transformation('my_format', 'edgelist')
def my_format_to_edgelist(obj, ctx):
    return convert_to_edgelist(obj)

@register_transformation('edgelist', 'my_format')
def edgelist_to_my_format(edgelist, ctx):
    return MyFormat.from_edgelist(edgelist)
```

### Performance Characteristics

- **Conversion costs**: Range from 0.1 (simple format conversions) to 1.0 (complex algorithms)
- **Path caching**: LRU cache (4096 entries) for computed paths
- **Optional result caching**: Can enable per-conversion if needed
- **Lazy loading**: Modules only load if dependencies are available

### Future Enhancements

Potential additions that could be made:

1. **More formats**: 
   - GraphML/GEXF files
   - JSON graph specification
   - DOT language (full graphviz)
   - Neo4j/graph database formats

2. **Optimizations**:
   - Result caching by default
   - Parallel conversion for large batches
   - Streaming conversions for huge graphs

3. **Additional context options**:
   - Node/edge attribute mapping rules
   - Filtering predicates
   - Transformation functions

### Files Created/Modified

#### New Files (8)
1. `linked/cast.py` - Core conversion system
2. `linked/datasrc/edgelist.py` - Edge list converters
3. `linked/datasrc/networkx_graphs.py` - NetworkX converters
4. `linked/datasrc/dataframes.py` - pandas converters
5. `linked/datasrc/adjacency.py` - Adjacency format converters
6. `linked/tests/test_cast.py` - Comprehensive tests
7. `CAST_README.md` - User documentation
8. `misc/cast_demo.py` - Interactive demo

#### Modified Files (3)
1. `linked/__init__.py` - Added cast API exports
2. `linked/datasrc/minidot.py` - Added cast registrations
3. `linked/datasrc/vectors.py` - Added cast registrations
4. `README.md` - Added cast system overview

### Conclusion

The graph conversion system provides a robust, extensible, and user-friendly way to work with multiple graph representations in Python. It follows established design patterns, has comprehensive test coverage, and integrates seamlessly with existing `linked` functionality while maintaining backward compatibility.
