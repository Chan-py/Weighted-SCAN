import math
from collections import deque

def run(G, eps=0.5, mu=2):
    # Identify core nodes
    cores = {u for u in G.nodes()
            if sum(is_eps_neighbor(G, u, v, eps)for v in G.neighbors(u)) >= mu}
    
    # Cluster expansion
    clusters = {}
    visited = set()
    cluster_id = 0
    for u in cores:
        if u in visited:
            continue
        cluster_id += 1
        clusters[cluster_id] = set()
        queue = deque([u])
        visited.add(u)
        while queue:
            x = queue.popleft()
            clusters[cluster_id].add(x)
            for y in G.neighbors(x):
                if y not in visited and y in cores and is_eps_neighbor(G, x, y, eps):
                    visited.add(y)
                    queue.append(y)
        # Include directly reachable non-core
        for member in list(clusters[cluster_id]):
            for y in G.neighbors(member):
                if y not in visited and is_eps_neighbor(G, member, y, eps):
                    clusters[cluster_id].add(y)
                    visited.add(y)
                    
    # Classify hubs and outliers
    all_clustered = set().union(*clusters.values()) if clusters else set()
    hubs, outliers = set(), set()
    for u in G.nodes():
        if u in all_clustered:
            continue
        connected = {cid for cid, members in clusters.items()
                     if any(is_eps_neighbor(G, u, v, eps) for v in members)}
        if len(connected) >= 2:
            hubs.add(u)
        else:
            outliers.add(u)
    return clusters, hubs, outliers


def weighted_structural_similarity(G, u, v):
    """
    Weighted Structural Similarity:
    s_str'(u,v) = ( w(u,v) if (u,v) edge exists else 0 )
                  + sum_{x in N(u)∩N(v)} min(w(u,x), w(v,x))
                  ---------------------------------------------------
                  sqrt( (sum_{x in N(u)} w(u,x)) * (sum_{y in N(v)} w(v,y)) )
    """
    # Common Neighbors
    neigh_u = set(G.neighbors(u))
    neigh_v = set(G.neighbors(v))
    common = neigh_u & neigh_v

    # direct edge weight (없으면 0)
    w_uv = G[u][v]['weight'] if G.has_edge(u, v) else 0.0

    # 공통 이웃에 대한 min-weight 합
    common_sum = sum(
        min(G[u][x]['weight'], G[v][x]['weight'])
        for x in common
    )

    # 분자: direct + common
    numer = w_uv + common_sum

    # 분모: 각 노드의 total strength
    su = sum(data['weight'] for _, _, data in G.edges(u, data=True))
    sv = sum(data['weight'] for _, _, data in G.edges(v, data=True))
    if su == 0 or sv == 0:
        return 0.0

    return numer / math.sqrt(su * sv)

def cosine_similarity(G, u, v):
    """
    Cosine Similarity:
    s_cos'(u,v) = (
        w(u,v)^2 if (u,v) edge exists else 0
        + sum_{x in N(u)∩N(v)} w(u,x)*w(v,x)
    ) / (
        sqrt(sum_{x in N(u)} w(u,x)^2)
        * sqrt(sum_{y in N(v)} w(v,y)^2)
    )
    """
    # 이웃 집합
    neigh_u = set(G.neighbors(u))
    neigh_v = set(G.neighbors(v))
    common = neigh_u & neigh_v

    # direct edge weight (없으면 0)
    w_uv = G[u][v]['weight'] if G.has_edge(u, v) else 0.0
    # 분자: direct 연결 기여 (w_uv^2) + 공통 이웃 기여
    numer = w_uv**2 + sum(
        G[u][x]['weight'] * G[v][x]['weight']
        for x in common
    )

    # 분모: 각 노드의 L2 노름
    norm_u = math.sqrt(sum(data['weight']**2 for _, _, data in G.edges(u, data=True)))
    norm_v = math.sqrt(sum(data['weight']**2 for _, _, data in G.edges(v, data=True)))
    if norm_u == 0 or norm_v == 0:
        return 0.0

    return numer / (norm_u * norm_v)

def weighted_jaccard_similarity(G, u, v):
    """
    Weighted Jaccard Similarity:
    s_J(u,v) = sum_x min(w(u,x), w(v,x))
               / sum_x max(w(u,x), w(v,x))
    """
    neigh_u = set(G.neighbors(u))
    neigh_v = set(G.neighbors(v))
    all_neigh = neigh_u | neigh_v
    if not all_neigh:
        return 0.0
    numer = 0.0
    denom = 0.0
    for x in all_neigh:
        wu = G[u][x]['weight'] if x in neigh_u else 0.0
        wv = G[v][x]['weight'] if x in neigh_v else 0.0
        numer += min(wu, wv)
        denom += max(wu, wv)
    if denom == 0:
        return 0.0
    return numer / denom

def is_eps_neighbor(G, u, v, eps):
    return cosine_similarity(G, u, v) >= eps