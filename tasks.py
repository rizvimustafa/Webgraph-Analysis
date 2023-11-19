import networkx as nx
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

G = nx.read_edgelist('web-Stanford.txt', create_using=nx.DiGraph)
# Find the largest weakly connected component
largest_weakly_connected = max(nx.weakly_connected_components(G), key=len)
G_wcc = G.subgraph(largest_weakly_connected)

# Find the largest strongly connected component
largest_strongly_connected = max(nx.strongly_connected_components(G), key=len)
G_scc = G.subgraph(largest_strongly_connected)

# Compute PageRank for the largest weakly connected component
pagerank_wcc = nx.pagerank(G_wcc)

# Compute PageRank for the largest strongly connected component
pagerank_scc = nx.pagerank(G_scc)

'''
print("PageRank for the largest Weakly Connected Component:")
for node, rank in pagerank_wcc.items():
    print(f"Node: {node}, PageRank: {rank}")

print("\nPageRank for the largest Strongly Connected Component:")
for node, rank in pagerank_scc.items():
    print(f"Node: {node}, PageRank: {rank}")
'''
# TASK 3! TASK 3! TASK 3! TASK 3! TASK 3! TASK 3! TASK 3! TASK 3! TASK 3! TASK 3! TASK 3! TASK 3! TASK 3!


num_nodes_wcc = len(largest_weakly_connected)
num_edges_wcc = G_wcc.number_of_edges()
# Generate a Random Graph
random_graph = nx.fast_gnp_random_graph(num_nodes_wcc, 0.00008, seed=1)
# Calculate 'm' for the BA graph to approximate the number of edges in the largest WCC
m = num_edges_wcc // num_nodes_wcc

# Generate a Barabasi-Albert Graph
ba_graph = nx.barabasi_albert_graph(num_nodes_wcc, m, seed=1)
# Compute PageRank for the Random Graph
pagerank_random_graph = nx.pagerank(random_graph)

# Compute PageRank for the Barabasi-Albert Graph
pagerank_ba_graph = nx.pagerank(ba_graph)

# Calculate the correlation between the original WCC PageRank and Random Graph PageRank
correlation_random = pearsonr(list(pagerank_wcc.values()), list(pagerank_random_graph.values()))[0]

# Calculate the correlation between the original WCC PageRank and BA Graph PageRank
correlation_ba = pearsonr(list(pagerank_wcc.values()), list(pagerank_ba_graph.values()))[0]

print("Correlation with Random Graph:", correlation_random)
print("Correlation with BA Graph:", correlation_ba)


# Convert the PageRank dictionaries to NumPy arrays
original_pagerank = np.array(list(pagerank_wcc.values())).reshape(1, -1)
random_pagerank = np.array(list(pagerank_random_graph.values())).reshape(1, -1)
ba_pagerank = np.array(list(pagerank_ba_graph.values())).reshape(1, -1)

# Calculate cosine similarity
cosine_similarity_random = cosine_similarity(original_pagerank, random_pagerank)[0][0]
cosine_similarity_ba = cosine_similarity(original_pagerank, ba_pagerank)[0][0]

print("Cosine Similarity with Random Graph:", cosine_similarity_random)
print("Cosine Similarity with BA Graph:", cosine_similarity_ba)


# Sort the nodes by PageRank score and select the top-K nodes
K = 10  # You can adjust K as needed
top_nodes_wcc = sorted(pagerank_wcc, key=pagerank_wcc.get, reverse=True)[:K]

# Create a subgraph of the top-K nodes
G_wcc_top = G_wcc.subgraph(top_nodes_wcc)

# Generate the Random Graph
random_graph = nx.fast_gnp_random_graph(len(G_wcc_top), 0.00008, seed=1)

# Generate the Barabasi-Albert Graph
m = random_graph.number_of_edges() // len(random_graph)
ba_graph = nx.barabasi_albert_graph(len(G_wcc_top), m, seed=1)


# Visualize the original largest WCC, Random Graph, and BA Graph
plt.figure(figsize=(12, 8))
plt.subplot(131)
nx.draw(G_wcc_top, with_labels=True, node_size=1000, font_size=10, node_color='lightblue', font_color='black')
plt.title("Original WCC")

plt.subplot(132)
nx.draw(random_graph, with_labels=False, node_size=1000, node_color='lightblue')
plt.title("Random Graph")

plt.subplot(133)
nx.draw(ba_graph, with_labels=False, node_size=1000, node_color='lightblue')
plt.title("BA Graph")

plt.show()

'''
# Get the number of nodes and edges in the largest WCC and SCC
num_nodes_wcc = len(largest_wcc_subgraph)
num_edges_wcc = largest_wcc_subgraph.size()

num_nodes_scc = len(largest_scc_subgraph)
num_edges_scc = largest_scc_subgraph.size()

print("Largest Weakly Connected Component:")
print(f"Number of nodes: {num_nodes_wcc}")
print(f"Number of edges: {num_edges_wcc}")

print("Largest Strongly Connected Component:")
print(f"Number of nodes: {num_nodes_scc}")
print(f"Number of edges: {num_edges_scc}")
'''
'''
import networkx as nx
import matplotlib.pyplot as plt

# Read the graph from the edgelist file
G = nx.read_edgelist('web-Stanford.txt', create_using=nx.DiGraph)

# Find the largest weakly connected component
largest_weakly_connected = max(nx.weakly_connected_components(G), key=len)
G_wcc = G.subgraph(largest_weakly_connected)

# Compute PageRank for the largest WCC
pagerank_wcc = nx.pagerank(G_wcc)

# Sort the nodes by PageRank score and select the top-K nodes
K = 10  # You can adjust K as needed
top_nodes_wcc = sorted(pagerank_wcc, key=pagerank_wcc.get, reverse=True)[:K]

# Create a subgraph of the top-K nodes
G_wcc_top = G_wcc.subgraph(top_nodes_wcc)

# Generate the Random Graph
random_graph = nx.fast_gnp_random_graph(len(G_wcc_top), 0.00008, seed=1)

# Generate the Barabasi-Albert Graph
m = random_graph.number_of_edges() // len(random_graph)
ba_graph = nx.barabasi_albert_graph(len(G_wcc_top), m, seed=1)

# Visualize the original largest WCC, Random Graph, and BA Graph
plt.figure(figsize=(12, 8))
plt.subplot(131)
nx.draw(G_wcc_top, with_labels=True, node_size=1000, font_size=10, node_color='lightblue', font_color='black')
plt.title("Original WCC")

plt.subplot(132)
nx.draw(random_graph, with_labels=False, node_size=1000, node_color='lightblue')
plt.title("Random Graph")

plt.subplot(133)
nx.draw(ba_graph, with_labels=False, node_size=1000, node_color='lightblue')
plt.title("BA Graph")

plt.show()

'''