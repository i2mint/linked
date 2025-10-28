"""
Example usage of linked.datasrc.vectors module.

This script demonstrates various graph construction methods
and their use cases.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/claude')

from linked import (
    knn_graph,
    mutual_knn_graph,
    epsilon_graph,
    adaptive_knn_graph,
    random_graph,
)


def example_basic_knn():
    """Basic k-NN graph construction."""
    print("\n" + "="*60)
    print("Example 1: Basic k-NN Graph")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    vectors = np.random.rand(200, 32)
    print(f"Data shape: {vectors.shape}")
    
    # Build k-NN graph
    graph = knn_graph(vectors, n_neighbors=10, approximate=False)
    print(f"Graph shape: {graph.shape}")
    print(f"Number of edges: {len(graph)}")
    print(f"Sample edges:\n{graph[:5]}")
    
    # Statistics
    avg_weight = np.mean(graph[:, 2])
    print(f"Average edge weight (distance): {avg_weight:.4f}")


def example_mutual_knn():
    """Mutual k-NN graph for robust structure."""
    print("\n" + "="*60)
    print("Example 2: Mutual k-NN Graph")
    print("="*60)
    
    np.random.seed(42)
    vectors = np.random.rand(150, 64)
    print(f"Data shape: {vectors.shape}")
    
    # Regular k-NN
    graph_regular = knn_graph(vectors, n_neighbors=10, approximate=False)
    print(f"Regular k-NN edges: {len(graph_regular)}")
    
    # Mutual k-NN (no connectivity)
    graph_mutual = mutual_knn_graph(
        vectors, 
        n_neighbors=10, 
        approximate=False,
        ensure_connectivity="none"
    )
    print(f"Mutual k-NN edges: {len(graph_mutual)}")
    print(f"Sparsity reduction: {len(graph_mutual)/len(graph_regular)*100:.1f}%")
    
    # Mutual k-NN with MST connectivity
    graph_mutual_mst = mutual_knn_graph(
        vectors,
        n_neighbors=10,
        approximate=False,
        ensure_connectivity="mst"
    )
    print(f"Mutual k-NN + MST edges: {len(graph_mutual_mst)}")


def example_epsilon_graph():
    """Epsilon-neighborhood graph."""
    print("\n" + "="*60)
    print("Example 3: Epsilon-Neighborhood Graph")
    print("="*60)
    
    np.random.seed(42)
    vectors = np.random.rand(100, 16)
    print(f"Data shape: {vectors.shape}")
    
    # Try different radii
    for radius in [0.3, 0.5, 0.7]:
        graph = epsilon_graph(vectors, radius=radius)
        print(f"Radius {radius}: {len(graph)} edges")


def example_adaptive_knn():
    """Adaptive k-NN with density scaling."""
    print("\n" + "="*60)
    print("Example 4: Adaptive k-NN Graph (UMAP-style)")
    print("="*60)
    
    np.random.seed(42)
    # Create data with varying density
    cluster1 = np.random.randn(50, 10) * 0.3 + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cluster2 = np.random.randn(50, 10) * 1.0 + [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    vectors = np.vstack([cluster1, cluster2])
    print(f"Data shape: {vectors.shape} (two clusters with different densities)")
    
    # Standard k-NN
    graph_standard = knn_graph(vectors, n_neighbors=8, approximate=False)
    
    # Adaptive k-NN
    graph_adaptive = adaptive_knn_graph(
        vectors,
        n_neighbors=8,
        local_connectivity=1,
        bandwidth=1.0,
        approximate=False
    )
    
    print(f"Standard k-NN: {len(graph_standard)} edges")
    print(f"Adaptive k-NN: {len(graph_adaptive)} edges")
    
    # Compare weight distributions
    print(f"Standard weights - mean: {np.mean(graph_standard[:, 2]):.4f}, "
          f"std: {np.std(graph_standard[:, 2]):.4f}")
    print(f"Adaptive weights - mean: {np.mean(graph_adaptive[:, 2]):.4f}, "
          f"std: {np.std(graph_adaptive[:, 2]):.4f}")


def example_random_graphs():
    """Random graph generation."""
    print("\n" + "="*60)
    print("Example 5: Random Graphs")
    print("="*60)
    
    n_nodes = 50
    
    # Erdos-Renyi
    graph_er = random_graph(n_nodes, model="erdos_renyi", p=0.1, seed=42)
    print(f"Erdos-Renyi (p=0.1): {len(graph_er)} edges")
    
    # Barabasi-Albert
    graph_ba = random_graph(n_nodes, model="barabasi_albert", m=2, seed=42)
    print(f"Barabasi-Albert (m=2): {len(graph_ba)} edges")
    
    # Watts-Strogatz
    graph_ws = random_graph(n_nodes, model="watts_strogatz", k=4, p=0.2, seed=42)
    print(f"Watts-Strogatz (k=4, p=0.2): {len(graph_ws)} edges")
    
    # Weighted
    graph_weighted = random_graph(
        n_nodes, 
        model="erdos_renyi", 
        p=0.1, 
        weighted=True, 
        seed=42
    )
    print(f"Weighted graph: mean weight = {np.mean(graph_weighted[:, 2]):.4f}")


def example_large_scale():
    """Large-scale graph construction."""
    print("\n" + "="*60)
    print("Example 6: Large-Scale Processing")
    print("="*60)
    
    import time
    
    # Generate larger dataset
    np.random.seed(42)
    n_samples = 5000
    n_features = 256
    vectors = np.random.rand(n_samples, n_features)
    print(f"Data shape: {vectors.shape}")
    
    # Exact k-NN (slower)
    start = time.time()
    graph_exact = knn_graph(vectors, n_neighbors=15, approximate=False, n_jobs=-1)
    time_exact = time.time() - start
    print(f"Exact k-NN: {time_exact:.2f} seconds, {len(graph_exact)} edges")
    
    # Approximate k-NN (faster, if pynndescent available)
    try:
        start = time.time()
        graph_approx = knn_graph(vectors, n_neighbors=15, approximate=True, n_jobs=-1)
        time_approx = time.time() - start
        print(f"Approximate k-NN: {time_approx:.2f} seconds, {len(graph_approx)} edges")
        print(f"Speedup: {time_exact/time_approx:.1f}x")
    except Exception as e:
        print(f"Approximate k-NN not available (pynndescent not installed)")


def example_metrics():
    """Different distance metrics."""
    print("\n" + "="*60)
    print("Example 7: Different Distance Metrics")
    print("="*60)
    
    np.random.seed(42)
    vectors = np.random.rand(100, 32)
    
    for metric in ["euclidean", "manhattan", "cosine"]:
        graph = knn_graph(vectors, n_neighbors=10, metric=metric, approximate=False)
        avg_weight = np.mean(graph[:, 2])
        print(f"{metric.capitalize()} metric: {len(graph)} edges, "
              f"avg distance = {avg_weight:.4f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LINKED.DATASRC.VECTORS - USAGE EXAMPLES")
    print("="*60)
    
    example_basic_knn()
    example_mutual_knn()
    example_epsilon_graph()
    example_adaptive_knn()
    example_random_graphs()
    example_large_scale()
    example_metrics()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
