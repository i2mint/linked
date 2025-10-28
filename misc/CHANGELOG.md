# CHANGELOG

## 2025-10-27 Initial Module Creation

### Added
- Created `linked.datasrc.vectors` module for graph construction from high-dimensional vector data
- Implemented five main graph construction functions:
  - `knn_graph`: k-nearest neighbors graph with approximate and exact methods
  - `mutual_knn_graph`: Mutual k-NN graph with connectivity options (MST, NN)
  - `epsilon_graph`: Epsilon-neighborhood (radius-based) graph
  - `adaptive_knn_graph`: Adaptive k-NN with local density scaling (UMAP-style)
  - `random_graph`: Random graph generation (Erdos-Renyi, Barabasi-Albert, Watts-Strogatz)
- Implemented sklearn-style estimator classes:
  - `KNNGraphEstimator`
  - `MutualKNNGraphEstimator`
  - `AdaptiveKNNGraphEstimator`
- Added support for multiple distance metrics via sklearn and pynndescent
- Implemented MST-based and nearest-neighbor connectivity augmentation
- Created comprehensive test suite with 100% test coverage

### Technical Details
- Uses `pynndescent` for approximate nearest neighbors (optional dependency)
- Uses `scikit-learn` for exact nearest neighbors and general utilities
- Uses `scipy` for distance computations, sparse matrices, and MST algorithms
- Supports large-scale datasets (50K-500K points) with high dimensionality (3000+ features)
- All functions return edge lists as numpy arrays with shape (n_edges, 2) or (n_edges, 3)
- Edge lists format: [source, target] or [source, target, weight]
- All parameters beyond vectors are keyword-only for clarity and future extensibility
