# Usage example:
#   python main.py --network ../dataset/email.edgelist \
#                  --eps 0.25 --mu 2 --similarity Gen

import argparse
import time
import networkx as nx

import clustering
import similarity

from metrics import compute_ARI, compute_modularity
from utils import load_ground_truth, plot_clusters

# --------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Weighted-SCAN runner")

parser.add_argument("--eps", type=float, default=0.5,
                    help="ε similarity threshold")
parser.add_argument("--mu",  type=int,   default=2,
                    help="minimum number of ε-neighbors (core)")
parser.add_argument("--gamma",  type=float,   default=1,
                    help="degree considering undirected edges")

parser.add_argument("--similarity", choices=["scan", "wscan", "cosine", "Gen", "Jaccard"],
                    default="Gen",
                    help="choose similarity function")
parser.add_argument("--network", default="../dataset/example.dat",
                    help="path to weighted edge list (u v w)")

args = parser.parse_args()

# --------------------------------------------------------------------
# load network  (expects 'u v weight' per line)
# --------------------------------------------------------------------
G = nx.read_weighted_edgelist(args.network, nodetype=int)

print(len(G.nodes))
print(len(G.edges))

print(f"Loaded graph: {args.network}  "
      f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

# --------------------------------------------------------------------
# load answer
# --------------------------------------------------------------------
# ground_truth = load_ground_truth("../dataset/real/LFR_labels.dat")

# --------------------------------------------------------------------
# run selected algorithm
# --------------------------------------------------------------------
sim = {"scan" : similarity.scan_similarity, "wscan" : similarity.wscan_similarity,
       "cosine" : similarity.cosine_similarity, "Gen" : similarity.Gen_wscan_similarity, 
       "Jaccard" : similarity.weighted_jaccard_similarity}

similarity_func = sim[args.similarity]

start = time.time()
clusters, hubs, outliers = clustering.run(G, similarity_func, eps=args.eps, mu=args.mu, gamma=args.gamma)
runtime = time.time() - start

# --------------------------------------------------------------------
# basic report
# --------------------------------------------------------------------
print("\n=== RESULT SUMMARY ===")
print(f"similarity        : {args.similarity}")
print(f"ε, μ              : {args.eps}, {args.mu}")
print(f"runtime (seconds) : {runtime:.3f}")
print(f"#clusters         : {len(clusters)}")
print(f"#hubs             : {len(hubs)}")
print(f"#outliers         : {len(outliers)}")

# for cid, nodes in list(clusters.items()):
#     print(f"  cluster {cid:<2} size={len(nodes)} nodes={list(nodes)}")

# --------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------

# Q = compute_modularity_with_outliers(G, clusters, hubs, outliers)
# print(f"Modularity including outliers = {Q:.4f}")
# Q = compute_modularity_clustered_only(G, clusters)
# print(f"Modularity for only clusters = {Q:.4f}")
# C = conductance(G, clusters)
# print(f"Conductance C = {C:.4f}")

# intra = intra_density(G, clusters)
# print(f"Intra = {intra:.4f}")
# inter = inter_density(G, clusters)
# print(f"Inter = {inter:.4f}")

# ari_score = compute_ARI(clusters, ground_truth)
# print(f"Adjusted Rand Index: {ari_score:.4f}")

# modularity = compute_modularity(G, clusters)
# print(f"Modularity: {modularity:.4f}")

plot_clusters(G, clusters)