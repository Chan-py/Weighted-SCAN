import networkx as nx
from itertools import combinations

import wscan
import scan

from utils import load_ground_truth
from check_cluster import compute_ARI, compute_modularity
from detect_optimal_parameter import detect_epsilon_first_knee, detect_epsilon_kneedle, suggest_mu

G = nx.read_weighted_edgelist("../dataset/real/les_miserable_edges.dat", nodetype=int)
# ground_truth = load_ground_truth("../dataset/real/LFR_labels.dat")

sim_funcs = [wscan.weighted_structural_similarity, wscan.cosine_similarity, wscan.weighted_jaccard_similarity, 
             scan.structural_similarity, wscan.wscan_tfp_similarity]
for_debugging = ["weighted_structural_similarity", "cosine_similarity", "weighted_jaccard_similarity", 
             "structural_similarity", "wscan_tfp_similarity"]
for i in range(2, 3):
    similarity_func = sim_funcs[i]

    sims = []
    for u, v in combinations(G.nodes(), 2):
        sims.append(similarity_func(G, u, v))

    sims = [s for s in sims if s > 0]

    candidates = []

    eps = detect_epsilon_first_knee(sims)
    mu = suggest_mu(G, eps, similarity_func)
    candidates.append((eps, mu))

    eps = detect_epsilon_kneedle(sims)
    mu = suggest_mu(G, eps, similarity_func)
    candidates.append((eps, mu))

    best_ari = -1.0
    best_mod = -1.0
    best_ari_params = (None, None)
    best_mod_params = (None, None)

    # 6) 그리드 서치
    for eps, mu in candidates:
        clusters, hubs, outliers = wscan.run(G, similarity_func, eps=eps, mu=mu)

        # ari = compute_ARI(clusters, ground_truth)
        # print(f"ε={eps:.2f}, μ={mu:2d} -> ARI={ari:.4f}")
        # if ari > best_ari:
        #     best_ari = ari
        #     best_ari_params = (eps, mu)
        modularity = compute_modularity(G, clusters)
        print(f"ε={eps:.2f}, μ={mu:2d} -> Modularity={modularity:.4f}")
        if modularity > best_mod:
            best_mod = modularity
            best_mod_params = (eps, mu)


    # 7) 결과 출력
    print("######", for_debugging[i])
    # eps_opt, mu_opt = best_ari_params
    # print("=== Best Parameters ===")
    # print(f"ε* = {eps_opt:.4f}, μ* = {mu_opt}, ARI* = {best_ari:.4f}")
    eps_opt, mu_opt = best_mod_params
    print(f"ε* = {eps_opt:.4f}, μ* = {mu_opt}, Modulairty* = {best_mod:.4f}")