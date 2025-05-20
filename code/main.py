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
# import wscan          # your Weighted-SCAN implementation
# import unweighted_scan # thin wrapper around original SCAN

# --------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Weighted-SCAN runner")

parser.add_argument("--eps", type=float, default=0.3,
                    help="ε similarity threshold")
parser.add_argument("--mu",  type=int,   default=2,
                    help="minimum number of ε-neighbors (core)")
parser.add_argument("--algorithm", choices=["weighted", "scan"],
                    default="scan",
                    help="choose algorithm variant")
parser.add_argument("--network", default="../dataset/real/email-core.dat",
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

print(f"Loaded graph: {args.network}  "
      f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

# --------------------------------------------------------------------
# run selected algorithm
# --------------------------------------------------------------------
start = time.time()

if args.algorithm == "weighted":
    clusters, hubs, outliers = wscan.run(G, eps=args.eps, mu=args.mu)
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

# optional: list first few clusters
for cid, nodes in list(clusters.items())[:3]:
    print(f"  cluster {cid:<2} size={len(nodes)} sample={list(nodes)[:5]}")