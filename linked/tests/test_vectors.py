"""
Tests for linked.datasrc.vectors module.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/claude')

from linked.datasrc.vectors import (
    knn_graph,
    mutual_knn_graph,
    epsilon_graph,
    adaptive_knn_graph,
    random_graph,
    KNNGraphEstimator,
    MutualKNNGraphEstimator,
    AdaptiveKNNGraphEstimator,
)


def test_knn_graph():
    """Test basic k-NN graph construction."""
    print("Testing knn_graph...")
    vectors = np.random.rand(100, 10)
    
    # Test with distance mode
    graph = knn_graph(vectors, n_neighbors=5, approximate=False)
    assert graph.shape[1] == 3, "Should have 3 columns (source, target, weight)"
    assert len(graph) == 100 * 5, "Should have n_samples * n_neighbors edges"
    
    # Test with connectivity mode
    graph = knn_graph(vectors, n_neighbors=5, mode="connectivity", approximate=False)
    assert graph.shape[1] == 2, "Should have 2 columns (source, target)"
    
    print("✓ knn_graph tests passed")


def test_mutual_knn_graph():
    """Test mutual k-NN graph construction."""
    print("Testing mutual_knn_graph...")
    vectors = np.random.rand(50, 8)
    
    graph = mutual_knn_graph(vectors, n_neighbors=5, approximate=False, ensure_connectivity="none")
    assert graph.shape[1] == 3, "Should have 3 columns"
    # Mutual graph should have fewer edges than regular k-NN
    assert len(graph) <= 50 * 5, "Mutual graph should be sparser"
    
    # Test with MST connectivity
    graph_mst = mutual_knn_graph(vectors, n_neighbors=5, approximate=False, ensure_connectivity="mst")
    assert len(graph_mst) >= len(graph), "MST version should have at least as many edges"
    
    print("✓ mutual_knn_graph tests passed")


def test_epsilon_graph():
    """Test epsilon-neighborhood graph construction."""
    print("Testing epsilon_graph...")
    vectors = np.random.rand(30, 5)
    
    # Use a radius that should capture some neighbors
    graph = epsilon_graph(vectors, radius=0.5)
    assert graph.shape[1] == 3, "Should have 3 columns"
    assert len(graph) >= 0, "Should have some edges or be empty"
    
    # Test connectivity mode
    graph_conn = epsilon_graph(vectors, radius=0.5, mode="connectivity")
    assert graph_conn.shape[1] == 2, "Should have 2 columns"
    
    print("✓ epsilon_graph tests passed")


def test_adaptive_knn_graph():
    """Test adaptive k-NN graph construction."""
    print("Testing adaptive_knn_graph...")
    vectors = np.random.rand(40, 6)
    
    graph = adaptive_knn_graph(vectors, n_neighbors=5, approximate=False)
    assert graph.shape[1] == 3, "Should have 3 columns with weights"
    assert len(graph) == 40 * 5, "Should have n_samples * n_neighbors edges"
    # Weights should be between 0 and 1 (after exponential)
    assert np.all(graph[:, 2] >= 0) and np.all(graph[:, 2] <= 1), "Weights should be in [0, 1]"
    
    print("✓ adaptive_knn_graph tests passed")


def test_random_graph():
    """Test random graph generation."""
    print("Testing random_graph...")
    
    # Test Erdos-Renyi
    graph = random_graph(20, model="erdos_renyi", p=0.3, seed=42)
    assert graph.shape[1] == 2, "Should have 2 columns"
    assert len(graph) > 0, "Should have some edges"
    
    # Test with weights
    graph_weighted = random_graph(20, model="erdos_renyi", p=0.3, weighted=True, seed=42)
    assert graph_weighted.shape[1] == 3, "Should have 3 columns with weights"
    
    # Test Barabasi-Albert
    graph_ba = random_graph(30, model="barabasi_albert", m=2, seed=42)
    assert len(graph_ba) > 0, "BA graph should have edges"
    
    # Test Watts-Strogatz
    graph_ws = random_graph(20, model="watts_strogatz", k=4, p=0.1, seed=42)
    assert len(graph_ws) > 0, "WS graph should have edges"
    
    print("✓ random_graph tests passed")


def test_estimators():
    """Test sklearn-style estimator classes."""
    print("Testing estimator classes...")
    vectors = np.random.rand(50, 8)
    
    # Test KNNGraphEstimator
    est = KNNGraphEstimator(n_neighbors=5, approximate=False)
    graph = est.fit_transform(vectors)
    assert graph.shape[1] == 3, "KNN estimator should return 3 columns"
    
    # Test MutualKNNGraphEstimator
    est_mutual = MutualKNNGraphEstimator(n_neighbors=5, approximate=False)
    graph_mutual = est_mutual.fit_transform(vectors)
    assert graph_mutual.shape[1] == 3, "Mutual KNN estimator should return 3 columns"
    
    # Test AdaptiveKNNGraphEstimator
    est_adaptive = AdaptiveKNNGraphEstimator(n_neighbors=5, approximate=False)
    graph_adaptive = est_adaptive.fit_transform(vectors)
    assert graph_adaptive.shape[1] == 3, "Adaptive KNN estimator should return 3 columns"
    
    print("✓ Estimator tests passed")


def test_edge_format():
    """Test that edge formats are correct."""
    print("Testing edge formats...")
    vectors = np.random.rand(20, 5)
    
    # All graphs should return numpy arrays
    graph = knn_graph(vectors, n_neighbors=3, approximate=False)
    assert isinstance(graph, np.ndarray), "Should return numpy array"
    
    # Check integer node indices
    assert np.issubdtype(graph[:, 0].dtype, np.integer) or np.all(graph[:, 0] == graph[:, 0].astype(int))
    assert np.issubdtype(graph[:, 1].dtype, np.integer) or np.all(graph[:, 1] == graph[:, 1].astype(int))
    
    # Check node indices are in valid range
    assert np.all(graph[:, 0] >= 0) and np.all(graph[:, 0] < len(vectors))
    assert np.all(graph[:, 1] >= 0) and np.all(graph[:, 1] < len(vectors))
    
    print("✓ Edge format tests passed")


if __name__ == "__main__":
    print("Running tests for linked.datasrc.vectors...")
    print("=" * 60)
    
    test_knn_graph()
    test_mutual_knn_graph()
    test_epsilon_graph()
    test_adaptive_knn_graph()
    test_random_graph()
    test_estimators()
    test_edge_format()
    
    print("=" * 60)
    print("All tests passed! ✓")
