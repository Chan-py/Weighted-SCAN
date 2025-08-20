import networkx as nx
from itertools import combinations

import similarity
import clustering

from utils import load_ground_truth, eps_grid_by_quantiles
from metrics import compute_ARI, compute_NMI, compute_modularity, compute_DBI, compute_SI, compute_Qs

# G = nx.read_weighted_edgelist("../dataset/real/ca-cit-HepTh/network.dat", nodetype=int)     # 22908       너무 오래 걸림
# G = nx.read_weighted_edgelist("../dataset/real/sociopatterns-infectious/network.dat", nodetype=int)   # 410 Pass
# G = nx.read_weighted_edgelist("../dataset/real/moreno_names/network.dat", nodetype=int)     # 1773         다 너무 낮음
# G = nx.read_weighted_edgelist("../dataset/real/dnc-corecipient/network.dat", nodetype=int)     # 906     다 너무 낮음 낫배드

G = nx.read_weighted_edgelist("../dataset/real/les_miserable/network.dat", nodetype=int)      # 58
# G = nx.read_weighted_edgelist("../dataset/real/collegemsg/network.dat", nodetype=int)         # 1203
# G = nx.read_weighted_edgelist("../dataset/real/collegemsg_edges.dat", nodetype=int) 

print(f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

sim = {"scan" : similarity.scan_similarity, "wscan" : similarity.wscan_similarity,
       "cosine" : similarity.cosine_similarity, "Gen" : similarity.Gen_wscan_similarity, 
       "Jaccard" : similarity.weighted_jaccard_similarity}

import numpy as np

for s in sim:
       similarity_func = sim[s]

       best_ari = -1
       best_ari_params = (0, 0)

       max_mod = -1.0
       max_parm_mod = (0, 0)

       # DBI: 낮을수록 좋음
       best_dbi = float('inf')
       best_parm_dbi = (0, 0)

       # Silhouette: 높을수록 좋음
       best_si = -1.0
       best_parm_si = (0, 0)

       # Similarity-based Modularity Qs: 높을수록 좋음
       best_qs = -1.0
       best_parm_qs = (0, 0)


       eps_candidates = np.arange(0.1, 0.95, 0.05)
       if s == "wscan":
              eps_candidates = eps_grid_by_quantiles(G, similarity_func)
       for e in eps_candidates:
              for mu in range(2, 5):
                     clusters, hubs, outliers = clustering.run(G, similarity_func, eps=e, mu=mu)

                     # ari = compute_ARI(clusters, ground_truth)
                     # # print(f"ε={eps:.2f}, μ={mu:2d} -> ARI={ari:.4f}")
                     # if ari > best_ari:
                     #        best_ari = ari
                     #        best_ari_params = (e, mu)

                     # Modularity
                     modularity = compute_modularity(G, clusters)
                     if modularity > max_mod:
                            max_mod = modularity
                            max_parm_mod = (e, mu)

                     # DBI (낮을수록 좋음)
                     dbi = compute_DBI(G, clusters)
                     if dbi < best_dbi:
                            best_dbi = dbi
                            best_parm_dbi = (e, mu)

                     # Silhouette (높을수록 좋음)
                     si = compute_SI(G, clusters)
                     if si > best_si:
                            best_si = si
                            best_parm_si = (e, mu)

                     # Similarity-based Modularity Qs (높을수록 좋음)
                     qs = compute_Qs(G, clusters, similarity_func, 1)
                     if qs > best_qs:
                            best_qs = qs
                            best_parm_qs = (e, mu)

                     # print(f"[{s}] ε={e:.2f} -> Mod={modularity:.4f}, DBI={dbi:.4f}, SI={si:.4f}, Qs={qs:.4f}")

       print("======== Max/Min over", s)
       print(f"(Mod) ε={max_parm_mod[0]:.2f}, μ={max_parm_mod[1]:2d} -> {max_mod:.4f}")
       print(f"(DBI) ε={best_parm_dbi[0]:.2f}, μ={best_parm_dbi[1]:2d} -> {best_dbi:.4f}  (lower is better)")
       print(f"(SI ) ε={best_parm_si[0]:.2f}, μ={best_parm_si[1]:2d} -> {best_si:.4f}")
       print(f"(Qs ) ε={best_parm_qs[0]:.2f}, μ={best_parm_qs[1]:2d} -> {best_qs:.4f}")
       print()
              