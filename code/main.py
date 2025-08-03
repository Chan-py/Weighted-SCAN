#!/usr/bin/env python3
# main.py – minimal driver for Weighted-SCAN experiments
#
# Usage example:
#   python main.py --network ../dataset/email.edgelist \
#                  --eps 0.25 --mu 2 --algorithm weighted
#   python main.py --algorithm unweighted      # for baseline

import argparse
import time
import networkx as nx

import scan
import wscan          # your Weighted-SCAN implementation

from check_cluster import compute_ARI, compute_modularity
from utils import load_ground_truth
import plot

# --------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Weighted-SCAN runner")

parser.add_argument("--eps", type=float, default=0.5,
                    help="ε similarity threshold")
parser.add_argument("--mu",  type=int,   default=2,
                    help="minimum number of ε-neighbors (core)")
parser.add_argument("--algorithm", choices=["wscan", "scan"],
                    default="wscan",
                    help="choose algorithm variant")
parser.add_argument("--network", default="../dataset/example.dat",
                    help="path to weighted edge list (u v w)")
parser.add_argument("--directed", action="store_true",
                    help="treat graph as directed")
args = parser.parse_args()

# --------------------------------------------------------------------
# load network  (expects 'u v weight' per line)
# --------------------------------------------------------------------
if args.directed:
    G = nx.read_weighted_edgelist(args.network,
                                  nodetype=int,
                                  create_using=nx.DiGraph)
else:
    G = nx.read_weighted_edgelist(args.network, nodetype=int)

print(len(G.nodes))
print(len(G.edges))

print(f"Loaded graph: {args.network}  "
      f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

# --------------------------------------------------------------------
# load answer
# --------------------------------------------------------------------
# ground_truth = load_ground_truth("../dataset/example.dat")

# --------------------------------------------------------------------
# run selected algorithm
# --------------------------------------------------------------------
similarity_func = wscan.weighted_structural_similarity
start = time.time()

if args.algorithm == "wscan":
    clusters, hubs, outliers = wscan.run(G, similarity_func, eps=args.eps, mu=args.mu)
else:  # unweighted baseline (weights ignored)
    clusters, hubs, outliers = scan.run(G, eps=args.eps, mu=args.mu)

runtime = time.time() - start

# --------------------------------------------------------------------
# basic report
# --------------------------------------------------------------------
print("\n=== RESULT SUMMARY ===")
print(f"algorithm        : {args.algorithm}")
print(f"ε, μ             : {args.eps}, {args.mu}")
print(f"runtime (seconds): {runtime:.3f}")
print(f"#clusters         : {len(clusters)}")
print(f"#hubs             : {len(hubs)}")
print(f"#outliers         : {len(outliers)}")

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

# plot.plot_clusters(G, clusters)

# optional: list first few clusters
for cid, nodes in list(clusters.items())[:3]:
    print(f"  cluster {cid:<2} size={len(nodes)} sample={list(nodes)}")