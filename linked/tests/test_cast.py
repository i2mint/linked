"""
Tests for linked.cast graph conversion system.
"""

from contextlib import suppress
import numpy as np
import sys

sys.path.insert(0, "/home/claude")


# Test basic imports
def test_imports():
    """Test that cast module can be imported."""
    print("Testing imports...")
    from linked import cast
    from linked.cast import graph_transformer, convert_graph, list_graph_kinds

    print("✓ Import tests passed")


def test_core_kinds():
    """Test that core graph kinds are registered."""
    print("\nTesting core kinds registration...")
    from linked.cast import list_graph_kinds

    kinds = list_graph_kinds()
    print(f"Registered kinds: {kinds}")

    # Core kinds should be present
    core_kinds = {
        'nodes_and_links',
        'edgelist',
        'weighted_edgelist',
        'minidot',
        'adjacency_matrix',
    }
    assert core_kinds.issubset(kinds), f"Missing core kinds: {core_kinds - kinds}"

    print("✓ Core kinds tests passed")


def test_minidot_conversions():
    """Test mini-dot conversions."""
    print("\nTesting mini-dot conversions...")
    from linked.cast import convert_graph

    # Test mini-dot -> nodes_and_links
    minidot_str = "1 -> 2\n2 -> 3"
    result = convert_graph(minidot_str, 'nodes_and_links', from_kind='minidot')

    assert 'nodes' in result, "Result should have 'nodes' key"
    assert 'links' in result, "Result should have 'links' key"
    assert len(result['nodes']) == 3, f"Should have 3 nodes, got {len(result['nodes'])}"
    assert len(result['links']) == 2, f"Should have 2 links, got {len(result['links'])}"

    print("✓ Mini-dot conversion tests passed")


def test_edgelist_conversions():
    """Test edge list conversions."""
    print("\nTesting edge list conversions...")
    from linked.cast import convert_graph

    # Create test edge list
    edgelist = np.array([[0, 1], [1, 2], [2, 0]])

    # Convert to nodes_and_links
    result = convert_graph(edgelist, 'nodes_and_links', from_kind='edgelist')

    assert 'nodes' in result, "Result should have nodes"
    assert 'links' in result, "Result should have links"
    assert len(result['nodes']) == 3, "Should have 3 nodes"
    assert len(result['links']) == 3, "Should have 3 links"

    # Test weighted edge list
    weighted_edgelist = np.array([[0, 1, 0.5], [1, 2, 0.8]])
    result = convert_graph(
        weighted_edgelist, 'nodes_and_links', from_kind='weighted_edgelist'
    )

    assert len(result['links']) == 2, "Should have 2 links"
    # Check that weights are preserved
    assert 'weight' in result['links'][0], "Links should have weight field"

    print("✓ Edge list conversion tests passed")


def test_adjacency_conversions():
    """Test adjacency matrix/list conversions."""
    print("\nTesting adjacency conversions...")
    from linked.cast import convert_graph

    # Create adjacency matrix
    adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # Convert to edge list
    edgelist = convert_graph(adj_matrix, 'edgelist', from_kind='adjacency_matrix')

    assert isinstance(edgelist, np.ndarray), "Should return numpy array"
    assert len(edgelist) > 0, "Should have edges"
    assert edgelist.shape[1] == 2, "Should be unweighted edgelist"

    # Test adjacency list
    adj_list = {0: [1, 2], 1: [2], 2: []}
    edgelist = convert_graph(adj_list, 'edgelist', from_kind='adjacency_list')

    assert isinstance(edgelist, np.ndarray), "Should return numpy array"
    assert len(edgelist) == 3, "Should have 3 edges"

    print("✓ Adjacency conversion tests passed")


def test_multi_hop_conversion():
    """Test multi-hop conversions through the graph."""
    print("\nTesting multi-hop conversions...")
    from linked.cast import convert_graph

    # Test mini-dot -> edgelist (should go through nodes_and_links)
    minidot_str = "1 -> 2\n2 -> 3"
    edgelist = convert_graph(minidot_str, 'edgelist', from_kind='minidot')

    assert isinstance(edgelist, np.ndarray), "Should return numpy array"
    assert len(edgelist) == 2, "Should have 2 edges"

    # Test edgelist -> adjacency_matrix
    adj_matrix = convert_graph(edgelist, 'adjacency_matrix')

    assert isinstance(adj_matrix, np.ndarray), "Should return numpy array"
    assert adj_matrix.ndim == 2, "Should be 2D"
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "Should be square"

    print("✓ Multi-hop conversion tests passed")


def test_networkx_conversions():
    """Test NetworkX conversions if available."""
    print("\nTesting NetworkX conversions...")

    try:
        import networkx as nx
        from linked.cast import convert_graph, list_graph_kinds

        kinds = list_graph_kinds()
        if 'networkx_graph' not in kinds:
            print("⊘ NetworkX kinds not registered (networkx may not be installed)")
            return

        # Create a NetworkX graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        # Convert to edge list
        edgelist = convert_graph(G, 'edgelist', from_kind='networkx_graph')

        assert isinstance(edgelist, np.ndarray), "Should return numpy array"
        assert len(edgelist) == 3, f"Should have 3 edges, got {len(edgelist)}"

        # Convert edge list back to NetworkX
        G2 = convert_graph(edgelist, 'networkx_graph', from_kind='edgelist')

        assert isinstance(G2, nx.Graph), "Should return NetworkX Graph"
        assert G2.number_of_edges() == 3, "Should have 3 edges"

        print("✓ NetworkX conversion tests passed")

    except ImportError:
        print("⊘ NetworkX not available, skipping NetworkX tests")


def test_pandas_conversions():
    """Test pandas DataFrame conversions if available."""
    print("\nTesting pandas DataFrame conversions...")

    try:
        import pandas as pd
        from linked.cast import convert_graph, list_graph_kinds

        kinds = list_graph_kinds()
        if 'edges_dataframe' not in kinds:
            print("⊘ DataFrame kinds not registered (pandas may not be installed)")
            return

        # Create edges DataFrame
        edges_df = pd.DataFrame({'source': [0, 1, 2], 'target': [1, 2, 0]})

        # Convert to edge list
        edgelist = convert_graph(edges_df, 'edgelist', from_kind='edges_dataframe')

        assert isinstance(edgelist, np.ndarray), "Should return numpy array"
        assert len(edgelist) == 3, "Should have 3 edges"

        # Convert edge list back to DataFrame
        df2 = convert_graph(edgelist, 'edges_dataframe', from_kind='edgelist')

        assert isinstance(df2, pd.DataFrame), "Should return DataFrame"
        assert len(df2) == 3, "Should have 3 rows"
        assert (
            'source' in df2.columns and 'target' in df2.columns
        ), "Should have source/target columns"

        print("✓ Pandas DataFrame conversion tests passed")

    except ImportError:
        print("⊘ Pandas not available, skipping DataFrame tests")


def test_vectors_conversions():
    """Test vectors -> graph conversions."""
    print("\nTesting vectors conversions...")
    from linked.cast import convert_graph, list_graph_kinds

    kinds = list_graph_kinds()
    if 'vectors' not in kinds:
        print("⊘ Vectors kind not registered")
        return

    # Create random vectors
    vectors = np.random.rand(20, 5)

    # Convert to weighted edge list using k-NN
    try:
        edgelist = convert_graph(
            vectors,
            'weighted_edgelist',
            from_kind='vectors',
            context={'n_neighbors': 3},
        )

        assert isinstance(edgelist, np.ndarray), "Should return numpy array"
        assert edgelist.shape[1] == 3, "Should be weighted (3 columns)"
        assert (
            len(edgelist) == 20 * 3
        ), f"Should have 60 edges (20 nodes * 3 neighbors), got {len(edgelist)}"

        print("✓ Vectors conversion tests passed")

    except ImportError as e:
        print(f"⊘ Vectors conversion failed (missing dependency): {e}")


def test_round_trip_conversions():
    """Test round-trip conversions (A -> B -> A)."""
    print("\nTesting round-trip conversions...")
    from linked.cast import convert_graph

    # Test edge list round trip
    original_edgelist = np.array([[0, 1], [1, 2], [2, 0]])

    # edgelist -> nodes_and_links -> edgelist
    intermediate = convert_graph(
        original_edgelist, 'nodes_and_links', from_kind='edgelist'
    )
    result = convert_graph(intermediate, 'edgelist', from_kind='nodes_and_links')

    assert isinstance(result, np.ndarray), "Should return numpy array"
    assert result.shape[1] == 2, "Should have 2 columns"

    # Test adjacency matrix round trip
    original_adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # adjacency_matrix -> edgelist -> adjacency_matrix
    edgelist = convert_graph(original_adj, 'edgelist', from_kind='adjacency_matrix')
    result_adj = convert_graph(edgelist, 'adjacency_matrix', from_kind='edgelist')

    assert isinstance(result_adj, np.ndarray), "Should return numpy array"
    assert result_adj.shape == original_adj.shape, "Should have same shape"

    print("✓ Round-trip conversion tests passed")


def test_context_parameters():
    """Test that context parameters are used correctly."""
    print("\nTesting context parameters...")
    from linked.cast import convert_graph

    # Test custom field names for nodes_and_links -> edgelist
    graph = {
        'nodes': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}],
        'links': [{'source': 'a', 'target': 'b'}, {'source': 'b', 'target': 'c'}],
    }

    edgelist = convert_graph(
        graph,
        'edgelist',
        from_kind='nodes_and_links',
        context={'id_field': 'id', 'source_field': 'source', 'target_field': 'target'},
    )

    assert isinstance(edgelist, np.ndarray), "Should return numpy array"
    assert len(edgelist) == 2, "Should have 2 edges"

    print("✓ Context parameter tests passed")


def test_auto_detection():
    """Test automatic kind detection."""
    print("\nTesting automatic kind detection...")
    from linked.cast import convert_graph

    # Test with edge list (should auto-detect)
    edgelist = np.array([[0, 1], [1, 2]])
    result = convert_graph(edgelist, 'nodes_and_links')  # No from_kind

    assert 'nodes' in result and 'links' in result, "Should detect edgelist and convert"

    # Test with nodes_and_links dict
    graph = {
        'nodes': [{'id': '0'}, {'id': '1'}],
        'links': [{'source': '0', 'target': '1'}],
    }
    edgelist_result = convert_graph(graph, 'edgelist')  # No from_kind

    assert isinstance(
        edgelist_result, np.ndarray
    ), "Should detect nodes_and_links and convert"

    print("✓ Auto-detection tests passed")


def test_sparse_adjacency():
    """Test sparse adjacency matrix conversions if scipy is available."""
    print("\nTesting sparse adjacency conversions...")

    try:
        from scipy import sparse
        from linked.cast import convert_graph, list_graph_kinds

        kinds = list_graph_kinds()
        if 'sparse_adjacency' not in kinds:
            print("⊘ Sparse adjacency kind not registered")
            return

        # Create sparse matrix
        row = np.array([0, 1, 2])
        col = np.array([1, 2, 0])
        data = np.array([1, 1, 1])
        sparse_adj = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

        # Convert to edge list
        edgelist = convert_graph(sparse_adj, 'edgelist', from_kind='sparse_adjacency')

        assert isinstance(edgelist, np.ndarray), "Should return numpy array"
        assert len(edgelist) > 0, "Should have edges"

        # Convert back to sparse
        sparse_result = convert_graph(
            edgelist, 'sparse_adjacency', from_kind='edgelist'
        )

        assert sparse.issparse(sparse_result), "Should return sparse matrix"

        print("✓ Sparse adjacency conversion tests passed")

    except ImportError:
        print("⊘ Scipy not available, skipping sparse adjacency tests")


def test_reachability():
    """Test reachability queries."""
    print("\nTesting reachability queries...")
    from linked.cast import reachable_from_kind, sources_for_kind

    # Test reachable from edgelist
    reachable = reachable_from_kind('edgelist')

    assert (
        'nodes_and_links' in reachable
    ), "Should be able to reach nodes_and_links from edgelist"
    assert (
        'adjacency_matrix' in reachable
    ), "Should be able to reach adjacency_matrix from edgelist"

    # Test sources for edgelist
    sources = sources_for_kind('edgelist')

    assert (
        'nodes_and_links' in sources
    ), "Should be able to convert from nodes_and_links to edgelist"
    assert (
        'adjacency_matrix' in sources
    ), "Should be able to convert from adjacency_matrix to edgelist"

    print("✓ Reachability tests passed")


if __name__ == "__main__":
    print("Running tests for linked.cast...")
    print("=" * 60)

    test_imports()
    test_core_kinds()
    test_minidot_conversions()
    test_edgelist_conversions()
    test_adjacency_conversions()
    test_multi_hop_conversion()
    test_networkx_conversions()
    test_pandas_conversions()
    test_vectors_conversions()
    test_round_trip_conversions()
    test_context_parameters()
    test_auto_detection()
    test_sparse_adjacency()
    test_reachability()

    print("=" * 60)
    print("All tests completed! ✓")
