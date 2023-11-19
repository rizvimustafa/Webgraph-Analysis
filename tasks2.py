import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the graph from the edge list file
G = nx.read_edgelist('web-Stanford.txt', create_using=nx.DiGraph)

#TASK 1

# Find the largest weakly connected component
largest_wcc = max(nx.weakly_connected_components(G), key=len)
largest_wcc_graph = G.subgraph(largest_wcc)

# Find the largest strongly connected component
largest_scc = max(nx.strongly_connected_components(G), key=len)
largest_scc_graph = G.subgraph(largest_scc)

# Report the number of nodes and edges in the largest WCC and SCC
num_nodes_wcc = largest_wcc_graph.number_of_nodes()
num_edges_wcc = largest_wcc_graph.number_of_edges()
num_nodes_scc = largest_scc_graph.number_of_nodes()
num_edges_scc = largest_scc_graph.number_of_edges()

'''
print("Largest Weakly Connected Component:")
print(f"Number of nodes: {num_nodes_wcc}")
print(f"Number of edges: {num_edges_wcc}")

print("\nLargest Strongly Connected Component:")
print(f"Number of nodes: {num_nodes_scc}")
print(f"Number of edges: {num_edges_scc}")
'''

#TASK 2

# Compute PageRank for the largest WCC
pagerank_wcc = nx.pagerank(largest_wcc_graph)

# Compute PageRank for the largest SCC
pagerank_scc = nx.pagerank(largest_scc_graph)
'''
print("PageRank for the Largest Weakly Connected Component:")
print(pagerank_wcc)

print("\nPageRank for the Largest Strongly Connected Component:")
print(pagerank_scc)
'''
#TASK 3

# Number of nodes in the largest WCC
n_wcc = largest_wcc_graph.number_of_nodes()

# Parameters for Random Graph
p_random = 0.00008
random_graph = nx.fast_gnp_random_graph(n_wcc, p_random, seed=1, directed=True)

# Parameters for Barabasi-Albert Graph
m = 2 * largest_wcc_graph.number_of_edges() // n_wcc
ba_graph = nx.barabasi_albert_graph(n_wcc, m, seed=1)

# Compute PageRank for the Random Graph
pagerank_random = nx.pagerank(random_graph)

# Compute PageRank for the Barabasi-Albert Graph
pagerank_ba = nx.pagerank(ba_graph)
'''
print("PageRank for the Random Graph:")
print(pagerank_random)

print("\nPageRank for the Barabasi-Albert Graph:")
print(pagerank_ba)
'''
#TASK 4

# Convert PageRank dictionaries to arrays
pagerank_wcc_array = np.array([pagerank_wcc[node] for node in largest_wcc])
pagerank_scc_array = np.array([pagerank_scc[node] for node in largest_scc])
pagerank_random_array = np.array([pagerank_random[node] for node in random_graph.nodes()])
pagerank_ba_array = np.array([pagerank_ba[node] for node in ba_graph.nodes()])

# Reshape arrays to have the same shape
pagerank_wcc_array = pagerank_wcc_array.reshape(1, -1)
pagerank_scc_array = pagerank_scc_array.reshape(1, -1)
pagerank_random_array = pagerank_random_array.reshape(1, -1)
pagerank_ba_array = pagerank_ba_array.reshape(1, -1)

# Calculate cosine similarity
cosine_similarity_wcc_random = cosine_similarity(pagerank_wcc_array, pagerank_random_array)
cosine_similarity_wcc_ba = cosine_similarity(pagerank_wcc_array, pagerank_ba_array)
'''
print("Cosine Similarity between WCC and Random Graph PageRank:", cosine_similarity_wcc_random[0][0])
print("Cosine Similarity between WCC and Barabasi-Albert Graph PageRank:", cosine_similarity_wcc_ba[0][0])
'''

#TASK 5

# Select the top-K nodes based on PageRank scores
K = 20  # Adjust K as needed

top_nodes_wcc = [node for node, _ in sorted(pagerank_wcc.items(), key=lambda x: -x[1])[:K]]
top_nodes_scc = [node for node, _ in sorted(pagerank_scc.items(), key=lambda x: -x[1])[:K]]

# Create subgraphs for the top-K nodes in WCC and SCC
subgraph_wcc = largest_wcc_graph.subgraph(top_nodes_wcc)
subgraph_scc = largest_scc_graph.subgraph(top_nodes_scc)

# Draw the subgraphs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
pos_wcc = nx.spring_layout(subgraph_wcc)
nx.draw(subgraph_wcc, pos_wcc, with_labels=True, node_size=100, font_size=8)
plt.title("Top-K Nodes in WCC")

plt.subplot(1, 2, 2)
pos_scc = nx.spring_layout(subgraph_scc)
nx.draw(subgraph_scc, pos_scc, with_labels=True, node_size=100, font_size=8)
plt.title("Top-K Nodes in SCC")

plt.show()
