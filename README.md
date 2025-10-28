# linked

Create and transform graphs

To install:	```pip install linked```


## mini_dot_to_graph_jdict

Make graphs from the (mini) dot language. 

```python
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

## linked.datasrc.vectors

Graph construction from high-dimensional vector data for network analysis and visualization.

This module provides tools to convert point/vector data into graph data (edge lists) suitable for network analysis, visualization, and dimensionality reduction. It focuses on sparse graph construction from large datasets (50K-500K points) with high dimensionality (up to 3000+ features).


### Quick Start

```python
import numpy as np
from linked import knn_graph, mutual_knn_graph, adaptive_knn_graph

# Generate sample data
vectors = np.random.rand(1000, 128)  # 1000 points, 128 dimensions

# Build a k-NN graph
graph = knn_graph(vectors, n_neighbors=15)
print(graph.shape)  # (15000, 3) - source, target, weight

# Build a mutual k-NN graph (more robust for high dimensions)
graph = mutual_knn_graph(vectors, n_neighbors=15, ensure_connectivity="mst")

# Build an adaptive k-NN graph (UMAP-style)
graph = adaptive_knn_graph(vectors, n_neighbors=15, local_connectivity=1)
```

### Graph Construction Methods

#### 1. k-Nearest Neighbors Graph

Connects each point to its k nearest neighbors.

```python
from linked import knn_graph

graph = knn_graph(
    vectors,
    n_neighbors=15,           # Number of neighbors per point
    metric="euclidean",       # Distance metric
    mode="distance",          # "distance" or "connectivity"
    include_self=False,       # Include self-loops
    approximate=True,         # Use approximate nearest neighbors
    n_jobs=-1                 # Parallel processing
)
```

**Output Format**: Array of shape (n_edges, 3) with columns [source, target, weight] or (n_edges, 2) for connectivity mode.

#### 2. Mutual k-Nearest Neighbors Graph

More robust for high-dimensional data. An edge exists only if both points are in each other's k-NN.

```python
from linked import mutual_knn_graph

graph = mutual_knn_graph(
    vectors,
    n_neighbors=15,
    metric="euclidean",
    mode="distance",
    approximate=True,
    ensure_connectivity="mst",  # "none", "mst", or "nn"
    n_jobs=-1
)
```

**Connectivity Options**:
- `"none"`: May result in disconnected components
- `"mst"`: Add minimum spanning tree edges to ensure connectivity
- `"nn"`: Connect isolated vertices to their nearest neighbor

#### 3. Epsilon-Neighborhood Graph

Connects all pairs of points within a radius.

```python
from linked import epsilon_graph

graph = epsilon_graph(
    vectors,
    radius=0.5,              # Maximum distance for connections
    metric="euclidean",
    mode="distance"
)
```

#### 4. Adaptive k-NN Graph

Uses local density scaling (UMAP-style) to adaptively weight edges based on local structure.

```python
from linked import adaptive_knn_graph

graph = adaptive_knn_graph(
    vectors,
    n_neighbors=15,
    metric="euclidean",
    local_connectivity=1,    # Min strong connections per point
    bandwidth=1.0,           # Gaussian kernel bandwidth
    approximate=True
)
```

#### 5. Random Graph Generation

Generate synthetic graphs for testing and benchmarking.

```python
from linked import random_graph

# Erdos-Renyi: random edges with probability p
graph = random_graph(n_nodes=100, model="erdos_renyi", p=0.1, seed=42)

# Barabasi-Albert: preferential attachment
graph = random_graph(n_nodes=100, model="barabasi_albert", m=3, seed=42)

# Watts-Strogatz: small-world network
graph = random_graph(n_nodes=100, model="watts_strogatz", k=6, p=0.1, seed=42)

# Add random weights
graph = random_graph(n_nodes=100, model="erdos_renyi", p=0.1, weighted=True)
```

### Sklearn-Style Estimators

For integration with sklearn pipelines:

```python
from linked import KNNGraphEstimator, MutualKNNGraphEstimator

# Create estimator
estimator = KNNGraphEstimator(n_neighbors=15, approximate=True)

# Fit and transform
graph = estimator.fit_transform(vectors)
```

Available estimators:
- `KNNGraphEstimator`
- `MutualKNNGraphEstimator`
- `AdaptiveKNNGraphEstimator`

### Distance Metrics

All functions support various distance metrics via sklearn:
- `"euclidean"` (default)
- `"manhattan"`
- `"cosine"`
- `"minkowski"`
- `"chebyshev"`
- And many more from `scipy.spatial.distance`

### Performance Considerations

#### Approximate vs. Exact

For large datasets, use `approximate=True` to enable approximate nearest neighbors via pynndescent:

```python
# Fast approximate search
graph = knn_graph(vectors, n_neighbors=15, approximate=True)

# Slower exact search
graph = knn_graph(vectors, n_neighbors=15, approximate=False)
```

Approximate methods are typically 10-100x faster with 90-99% accuracy.

#### Parallel Processing

Use `n_jobs=-1` to utilize all CPU cores:

```python
graph = knn_graph(vectors, n_neighbors=15, n_jobs=-1)
```

#### Memory Management

For very large datasets (>500K points), consider:
1. Using sparse output formats
2. Processing in batches
3. Using approximate methods
4. Reducing `n_neighbors`

### Output Format

All graph construction functions return numpy arrays:

**With weights** (mode="distance"):
```python
array([[0, 5, 0.342],    # source=0, target=5, weight=0.342
       [0, 8, 0.521],    # source=0, target=8, weight=0.521
       [1, 3, 0.234],    # source=1, target=3, weight=0.234
       ...])
```

**Without weights** (mode="connectivity"):
```python
array([[0, 5],    # source=0, target=5
       [0, 8],    # source=0, target=8
       [1, 3],    # source=1, target=3
       ...])
```

### Integration with Visualization

Example: Use with force-directed layout:

```python
import numpy as np
from linked import adaptive_knn_graph
import networkx as nx
import matplotlib.pyplot as plt

# Build graph
vectors = np.random.rand(100, 50)
edges = adaptive_knn_graph(vectors, n_neighbors=10)

# Convert to NetworkX
G = nx.Graph()
for source, target, weight in edges:
    G.add_edge(int(source), int(target), weight=weight)

# Layout and visualize
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=50, width=0.5)
plt.show()
```

### Best Practices

1. **High-dimensional data**: Use `mutual_knn_graph` or `adaptive_knn_graph` for better structure preservation

2. **Sparse graphs**: Start with smaller `n_neighbors` (5-15) and increase if needed

3. **Connectivity**: Use `ensure_connectivity="mst"` for visualization tasks

4. **Large datasets**: Enable `approximate=True` and set appropriate `n_jobs`

5. **Distance metrics**: Choose based on your data:
   - Euclidean: General-purpose, works well for normalized data
   - Cosine: Good for high-dimensional sparse data (e.g., text embeddings)
   - Manhattan: Robust to outliers

### References

The algorithms implemented are based on recent research:
- McInnes et al. (2018): UMAP - Uniform Manifold Approximation and Projection
- Tang et al. (2016): LargeVis - Visualizing Large-scale High-dimensional Data
- Jarvis & Patrick (1973): Clustering Using a Similarity Measure Based on Shared Near Neighbors
- Belkin & Niyogi (2003): Laplacian Eigenmaps

For more details, see the included research document.
