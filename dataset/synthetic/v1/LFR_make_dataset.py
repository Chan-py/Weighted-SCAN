# make_lfr_datasets.py
import random
import numpy as np
import networkx as nx
from typing import List, Tuple

# -------------------------
# 1) LFR 생성 + 기본 가중치
# -------------------------
def lfr_weighted(
    n=500,
    tau1=2.5, tau2=1.5,
    mu=0.3,
    avg_deg=12,
    min_comm=20,
    seed=42,
    base_weight_dist=("lognormal", {"mean":0.0, "sigma":1.0}),
) -> nx.Graph:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # NetworkX 버전별 인자 차이 대응
    try:
        G = nx.LFR_benchmark_graph(
            n, tau1, tau2, mu,
            average_degree=avg_deg,
            min_community=min_comm,
            seed=seed,
        )
    except TypeError:
        G = nx.LFR_benchmark_graph(
            n, tau1, tau2, mu,
            min_degree=max(2, int(avg_deg*0.8)),
            min_community=min_comm,
            seed=seed,
        )

    # 기본 가중치 부여
    for u, v in G.edges():
        if base_weight_dist[0] == "lognormal":
            m, s = base_weight_dist[1]["mean"], base_weight_dist[1]["sigma"]
            w = float(np.random.lognormal(m, s, 1)[0])
        elif base_weight_dist[0] == "pareto":
            a = base_weight_dist[1].get("alpha", 2.0)
            w = float(1.0 + rng.pareto(a))
        else:
            w = 1.0
        G[u][v]["weight"] = max(0.01, w)

    return G

# -------------------------
# 2) 커뮤니티 유틸
# -------------------------
def communities_of(G: nx.Graph) -> List[List[int]]:
    comms = {}
    for u, cset in G.nodes(data="community"):
        # LFR은 set/frozenset일 수 있음 -> 대표 하나 뽑기
        c = list(cset)[0] if isinstance(cset, (set, frozenset)) else cset
        comms.setdefault(c, []).append(u)
    return list(comms.values())

# -------------------------
# 3) 코사인-불리 케이스 주입
# -------------------------
def scale_shift_one_community(G, comm_nodes, factor=3.0):
    cset = set(comm_nodes)
    for u in comm_nodes:
        for v in G[u]:
            if v in cset:
                G[u][v]["weight"] *= factor

def add_few_heavy_bridges(G, commA, commB, k=5, heavy_w=50.0, seed=0):
    rng = random.Random(seed)
    A, B = list(commA), list(commB)
    for _ in range(k):
        u = rng.choice(A)
        v = rng.choice(B)
        if G.has_edge(u, v):
            G[u][v]["weight"] = max(G[u][v]["weight"], heavy_w)
        else:
            G.add_edge(u, v, weight=heavy_w)

def sprinkle_many_weak_intra(G, comm_nodes, p=0.15, weak_w=0.1, seed=0):
    rng = random.Random(seed)
    nodes = list(comm_nodes)
    n = len(nodes)
    trials = int(p * n * (n - 1) / 2)
    for _ in range(trials):
        u, v = rng.sample(nodes, 2)
        if G.has_edge(u, v):
            G[u][v]["weight"] = min(G[u][v]["weight"], weak_w)
        else:
            G.add_edge(u, v, weight=weak_w)

def make_dataset_case(case="scale_shift", seed=7) -> Tuple[nx.Graph, List[List[int]]]:
    G = lfr_weighted(seed=seed)
    comms = communities_of(G)
    comms.sort(key=len, reverse=True)
    if len(comms) < 2:
        return G, comms

    if case == "scale_shift":
        scale_shift_one_community(G, comms[0], factor=3.0)

    elif case == "weak_ties_heavy_bridges":
        sprinkle_many_weak_intra(G, comms[0], p=0.25, weak_w=0.1, seed=seed)
        add_few_heavy_bridges(G, comms[0], comms[1], k=8, heavy_w=80.0, seed=seed)

    elif case == "hub_hetero":
        # 허브의 내부 엣지는 매우 약하게, 비허브 내부는 중간 강화
        cset = set(comms[0])
        degs = sorted([(u, G.degree(u)) for u in comms[0]], key=lambda x: x[1], reverse=True)
        hubs = [u for u, _ in degs[: max(3, len(degs)//20) ]]
        for u in hubs:
            for v in G[u]:
                if v in cset:
                    G[u][v]["weight"] = min(G[u][v]["weight"], 0.05)
        for u in comms[0]:
            if u in hubs: 
                continue
            for v in G[u]:
                if v in cset:
                    G[u][v]["weight"] = max(G[u][v]["weight"], 2.0)

    return G, comms

# -------------------------
# 4) .dat & 라벨 파일 저장
# -------------------------
def write_dat(G: nx.Graph, path: str, start_index: int = 1, header: bool = False):
    """
    .dat 포맷: 한 줄에 `u v w` (공백 구분, 실수 가중치).
    start_index=1 -> 1-based 저장(실험 코드가 1-based면 이 옵션 유지).
    header=True 로 하면 맨 윗줄에 "# u v w" 헤더를 덧붙임.
    """
    mapping = {u: i+start_index for i, u in enumerate(G.nodes())}
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("# u v w\n")
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1.0)
            f.write(f"{mapping[u]} {mapping[v]} {w}\n")

def write_labels(G: nx.Graph, comms: List[List[int]], path: str, start_index: int = 1):
    """
    라벨 파일 포맷: `node label` (공백 구분, 1-based 노드/라벨).
    겹치는 커뮤니티가 있어도 대표 하나만 기록(LFR 기본은 non-overlap).
    """
    node_to_idx = {u: i+start_index for i, u in enumerate(G.nodes())}
    # 커뮤니티 id도 1-based로
    with open(path, "w", encoding="utf-8") as f:
        for cid, members in enumerate(comms, start=1):
            for u in members:
                f.write(f"{node_to_idx[u]} {cid}\n")

# -------------------------
# 5) 예시 실행
# -------------------------
if __name__ == "__main__":
    cases = [
        ("scale_shift", 1),
        ("weak_ties_heavy_bridges", 2),
        ("hub_hetero", 3),
    ]

    for name, seed in cases:
        G, comms = make_dataset_case(name, seed=seed)
        write_dat(G, f"{name}.dat", start_index=1, header=False)
        write_labels(G, comms, f"{name}.labels", start_index=1)
        print(f"Saved: {name}.dat, {name}.labels")
