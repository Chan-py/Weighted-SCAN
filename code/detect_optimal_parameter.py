import math
import networkx as nx
import numpy as np
from scipy.interpolate import UnivariateSpline

from itertools import combinations

from wscan import weighted_structural_similarity, cosine_similarity, weighted_jaccard_similarity, wscan_tfp_similarity
from scan import structural_similarity

def detect_epsilon_kneedle(sim_values):
    sims = sorted(sim_values, reverse=True)     # 1) 내림차순
    n     = len(sims)
    xs    = np.linspace(0, 1, n)                # 0 → 1
    ys    = np.array(sims)                      # 1 → 0

    # 2) 직선 y = 1 - x 와의 '아래쪽' 거리 (양수)
    dist  = (1 - xs) - ys                       # == y_line - y_curve

    # 3) 최대 거리 지점이 knee
    idx   = int(dist.argmax())
    return sims[idx]

def detect_epsilon_first_knee(sim_values):
    sims = sorted(sim_values, reverse=True)
    n     = len(sims)
    xs    = np.linspace(0, 1, n)
    ys    = np.array(sims)
    diff  = (1 - xs) - ys            # 직선보다 아래쪽 거리

    # ① 양→음으로 처음 바뀌는 지점(첫 로컬 최대) 탐색
    for i in range(n-2, 0, -1):
        if diff[i] > diff[i - 1] and diff[i] > diff[i + 1]:
            return sims[i]
    # ② 없으면 기존 방식 fallback
    return sims[int(diff.argmax())]


def suggest_mu(G, epsilon, sim_func):
    counts = [
        sum(1 for v in G.neighbors(u) if sim_func(G, u, v) >= epsilon)
        for u in G.nodes()
    ]
    # 평균보다 중앙값이 이상치에 덜 민감
    return max(1, int(round(np.median(counts))))

if __name__ == "__main__":
    G = nx.read_weighted_edgelist("../dataset/real/LFR_edges.dat")
    similarity_func = wscan_tfp_similarity

    sims = []
    for u, v in combinations(G.nodes(), 2):
        sims.append(similarity_func(G, u, v))

    sims = [s for s in sims if s > 0]

    sims_sorted = sorted(sims)

    print("#1 detect first kneedle")
    # ε 추정
    eps = detect_epsilon_first_knee(sims)
    print(f"Suggested epsilon (knee): {eps:.4f}")

    # μ 추정
    mu = suggest_mu(G, eps, similarity_func)
    print(f"Suggested mu (avg #ε-neighbors): {mu}")

    print("#2 detect maximum diff kneedle")

    eps = detect_epsilon_kneedle(sims)
    print(f"Suggested epsilon (knee): {eps:.4f}")

    mu = suggest_mu(G, eps, similarity_func)
    print(f"Suggested mu (avg #ε-neighbors): {mu}")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(sims_sorted)
    plt.xlabel('Ranked Pair Index')
    plt.ylabel('Similarity Value')
    plt.title('Similarity Distribution Across Node Pairs')
    plt.show()