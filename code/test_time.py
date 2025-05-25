import networkx as nx
from itertools import combinations
import time

import wscan
import scan

from utils import load_ground_truth
from check_cluster import compute_ARI, compute_modularity
from detect_optimal_parameter import detect_epsilon_first_knee, detect_epsilon_kneedle, suggest_mu

times = [1000, 2000, 5000, 10000, 20000]
for t in times:
    edges_file = "../dataset/real/running_time/" + str(t) + "edges.dat"
    labels_file = "../dataset/real/running_time/" + str(t) + "labels.dat"
    G = nx.read_weighted_edgelist(edges_file, nodetype=int)
    ground_truth = load_ground_truth(labels_file)

    sim_funcs = [wscan.weighted_structural_similarity, wscan.cosine_similarity, wscan.weighted_jaccard_similarity, 
                scan.structural_similarity, wscan.wscan_tfp_similarity]
    for_debugging = ["weighted_structural_similarity", "cosine_similarity", "weighted_jaccard_similarity", 
                "structural_similarity", "wscan_tfp_similarity"]
    for i in range(5):
        similarity_func = sim_funcs[i]

        sims = []
        for u, v in combinations(G.nodes(), 2):
            sims.append(similarity_func(G, u, v))

        sims = [s for s in sims if s > 0]

        candidates = []

        start_1 = time.time()
        eps = detect_epsilon_first_knee(sims)
        mu = suggest_mu(G, eps, similarity_func)
        candidates.append((eps, mu))

        eps = detect_epsilon_kneedle(sims)
        mu = suggest_mu(G, eps, similarity_func)
        candidates.append((eps, mu))

        print("time: ", t, ", similarity func: ", for_debugging[i])
        print("parameter detection")
        print(time.time() - start_1)

        best_ari = -1.0
        best_mod = -1.0
        best_ari_params = (None, None)
        best_mod_params = (None, None)

        # 6) 그리드 서치
        for eps, mu in candidates:
            start_2 = time.time()
            clusters, hubs, outliers = wscan.run(G, similarity_func, eps=eps, mu=mu)
            print("algorithm running time")
            print(time.time() - start_2)