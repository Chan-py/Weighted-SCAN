import networkx as nx
import numpy as np
from networkx.generators.community import LFR_benchmark_graph

# 1. 파라미터 설정
n = 500
tau1 = 3.0
tau2 = 1.5
mu = 0.1
avg_degree = 10
max_degree = 50
min_community = 20
seed = 42

# 2. LFR 그래프 생성 및 정리
G = LFR_benchmark_graph(
    n, tau1, tau2, mu,
    average_degree=avg_degree,
    max_degree=max_degree,
    min_community=min_community,
    seed=seed
)
G = nx.Graph(G)
G.remove_edges_from(nx.selfloop_edges(G))

# 3. community ground truth 추출 및 ID 부여
communities = {frozenset(G.nodes[v]['community']) for v in G.nodes()}
ground_truth = [set(c) for c in communities]
comm_id = {node: idx for idx, comm in enumerate(ground_truth) for node in comm}

# 4. 가중치 분포 설정 (예시)
intra_weight_range = (0.7, 1.0)   # 같은 커뮤니티 내부 엣지에서 샘플
inter_weight_range = (0.0, 0.3)   # 다른 커뮤니티 간 엣지에서 샘플

# 5. 엣지에 커뮤니티 기반 가중치 할당
for u, v, data in G.edges(data=True):
    if comm_id[u] == comm_id[v]:
        w = np.random.uniform(*intra_weight_range)
    else:
        w = np.random.uniform(*inter_weight_range)
    data['weight'] = w

# 6. edges.dat 쓰기: "u v weight"
with open("edges.dat", "w") as f:
    for u, v, data in G.edges(data=True):
        f.write(f"{u} {v} {data['weight']:.6f}\n")

# 7. labels.dat 쓰기: "u community_id"
with open("labels.dat", "w") as f:
    for node in G.nodes():
        f.write(f"{node} {comm_id[node]}\n")

print("→ edges.dat 와 labels.dat 저장 완료")
