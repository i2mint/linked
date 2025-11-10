"""
Example: Graph conversion system demonstration

This script demonstrates the various graph conversion capabilities
of the linked.cast module.
"""

import numpy as np
from linked import convert_graph, list_graph_kinds, reachable_from_kind

print("=" * 70)
print("Graph Conversion System Demo")
print("=" * 70)

# 1. Show available graph kinds
print("\n1. Available graph kinds:")
print("-" * 70)
kinds = list_graph_kinds()
for kind in sorted(kinds):
    print(f"  - {kind}")

# 2. Basic conversions
print("\n2. Basic conversions:")
print("-" * 70)

# Edge list to nodes_and_links
edgelist = np.array([[0, 1], [1, 2], [2, 0]])
print(f"Original edge list:\n{edgelist}")

graph = convert_graph(edgelist, 'nodes_and_links')
print(f"\nConverted to nodes_and_links:")
print(f"  Nodes: {len(graph['nodes'])} nodes")
print(f"  Links: {len(graph['links'])} links")
print(f"  First link: {graph['links'][0]}")

# 3. Mini-dot conversions
print("\n3. Mini-dot conversions:")
print("-" * 70)

minidot = """
A -> B, C
B -> D
C, D -> E
"""
print(f"Mini-dot input:\n{minidot}")

graph = convert_graph(minidot.strip(), 'nodes_and_links', from_kind='minidot')
print(f"Converted to nodes_and_links:")
print(f"  Nodes: {[n['id'] for n in graph['nodes']]}")
print(f"  Number of links: {len(graph['links'])}")

# 4. Multi-hop conversion
print("\n4. Multi-hop conversion (mini-dot -> adjacency matrix):")
print("-" * 70)

minidot_simple = "0 -> 1\n1 -> 2\n2 -> 0"
print(f"Mini-dot: {repr(minidot_simple)}")

adj_matrix = convert_graph(minidot_simple, 'adjacency_matrix', from_kind='minidot')
print(f"Adjacency matrix:\n{adj_matrix}")

# 5. Weighted graphs
print("\n5. Weighted edge lists:")
print("-" * 70)

weighted_edges = np.array([[0, 1, 0.5], [1, 2, 0.8], [2, 0, 0.3]])
print(f"Weighted edge list:\n{weighted_edges}")

graph = convert_graph(weighted_edges, 'nodes_and_links', from_kind='weighted_edgelist')
print(f"Converted to nodes_and_links (preserves weights):")
print(f"  First link with weight: {graph['links'][0]}")

# 6. Adjacency list
print("\n6. Adjacency list conversion:")
print("-" * 70)

adj_list = {0: [1, 2], 1: [2], 2: [0]}
print(f"Adjacency list: {adj_list}")

edgelist = convert_graph(adj_list, 'edgelist', from_kind='adjacency_list')
print(f"Converted to edge list:\n{edgelist}")

# 7. Vectors to graph (k-NN)
print("\n7. Vector similarity graph (k-NN):")
print("-" * 70)

np.random.seed(42)
vectors = np.random.rand(10, 5)
print(f"Vectors shape: {vectors.shape}")

knn_edges = convert_graph(
    vectors,
    'weighted_edgelist',
    from_kind='vectors',
    context={'n_neighbors': 3, 'metric': 'euclidean'},
)
print(f"k-NN graph (k=3):")
print(f"  Number of edges: {len(knn_edges)}")
print(f"  First 3 edges (with distances):\n{knn_edges[:3]}")

# 8. Show reachability
print("\n8. Reachability from 'edgelist':")
print("-" * 70)

reachable = reachable_from_kind('edgelist')
print(f"Can convert from edgelist to:")
for kind in sorted(reachable):
    print(f"  - {kind}")

# 9. Round-trip conversion
print("\n9. Round-trip conversion test:")
print("-" * 70)

original = np.array([[0, 1], [1, 2], [2, 3]])
print(f"Original edge list:\n{original}")

# edgelist -> nodes_and_links -> edgelist
intermediate = convert_graph(original, 'nodes_and_links')
result = convert_graph(intermediate, 'edgelist')
print(f"After round trip (edgelist -> nodes_and_links -> edgelist):\n{result}")

# 10. Context parameters
print("\n10. Custom context parameters:")
print("-" * 70)

# Custom field names
graph = {
    'nodes': [{'node_id': 'A'}, {'node_id': 'B'}, {'node_id': 'C'}],
    'links': [
        {'from': 'A', 'to': 'B', 'distance': 1.5},
        {'from': 'B', 'to': 'C', 'distance': 2.0},
    ],
}

context = {'id_field': 'node_id', 'source_field': 'from', 'target_field': 'to'}

edgelist = convert_graph(graph, 'edgelist', context=context)
print(f"Custom field names handled correctly:")
print(f"  Converted edge list shape: {edgelist.shape}")

print("\n" + "=" * 70)
print("Demo completed successfully!")
print("=" * 70)
