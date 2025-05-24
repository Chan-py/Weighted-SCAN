import networkx as nx

import wscan
import scan

from utils import load_ground_truth
from check_cluster import compute_ARI, compute_modularity

G = nx.read_weighted_edgelist("../dataset/real/collegemsg_edges.dat", nodetype=int)
ground_truth = load_ground_truth("../dataset/real/butterfly_labels.dat")

# ———————————————————————
# 5) ε, μ 후보 설정
eps_candidates = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
mu_candidates  = [2, 4, 6, 8, 10]

# best_ari = -1.0
best_mod = -1.0
best_params = (None, None)

# 6) 그리드 서치
for eps in eps_candidates:
    for mu in mu_candidates:
        clusters, hubs, outliers = scan.run(G, eps=eps, mu=mu)

        # ari = compute_ARI(clusters, ground_truth)
        # print(f"ε={eps:.2f}, μ={mu:2d} -> ARI={ari:.4f}")
        # if ari > best_ari:
        #     best_ari = ari
        #     best_params = (eps, mu)
        modularity = compute_modularity(G, clusters)
        print(f"ε={eps:.2f}, μ={mu:2d} -> Modularity={modularity:.4f}")
        if modularity > best_mod:
            best_mod = modularity
            best_params = (eps, mu)


# 7) 결과 출력
eps_opt, mu_opt = best_params
print("\n=== Best Parameters ===")
# print(f"ε* = {eps_opt:.2f}, μ* = {mu_opt}, ARI* = {best_ari:.4f}")
print(f"ε* = {eps_opt:.2f}, μ* = {mu_opt}, Modulairty* = {best_mod:.4f}")