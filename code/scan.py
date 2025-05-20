import math

def structural_similarity(G, u, v):
    neigh_u = set(G.neighbors(u)) | {u}
    neigh_v = set(G.neighbors(v)) | {v}
    common = neigh_u & neigh_v
    if not common:
        return 0.0
    return len(common) / math.sqrt(len(neigh_u) * len(neigh_v))

def run(G, eps=0.5, mu=2):
    # Identify core nodes
    cores = {u for u in G.nodes()
             if sum(structural_similarity(G, u, v) >= eps
                    for v in G.neighbors(u)) >= mu}
    # Cluster expansion
    clusters = {}
    visited = set()
    cluster_id = 0
    for u in cores:
        if u in visited:
            continue
        cluster_id += 1
        clusters[cluster_id] = set()
        queue = [u]
        visited.add(u)
        while queue:
            x = queue.pop(0)
            clusters[cluster_id].add(x)
            for y in G.neighbors(x):
                if y not in visited and y in cores and structural_similarity(G, x, y) >= eps:
                    visited.add(y)
                    queue.append(y)
        # Include directly reachable non-core
        for member in list(clusters[cluster_id]):
            for y in G.neighbors(member):
                if y not in visited and structural_similarity(G, member, y) >= eps:
                    clusters[cluster_id].add(y)
                    visited.add(y)
    # Classify hubs and outliers
    all_clustered = set().union(*clusters.values()) if clusters else set()
    hubs, outliers = set(), set()
    for u in G.nodes():
        if u in all_clustered:
            continue
        connected = {cid for cid, members in clusters.items()
                     if any(structural_similarity(G, u, v) >= eps for v in members)}
        if len(connected) >= 2:
            hubs.add(u)
        else:
            outliers.add(u)
    return clusters, hubs, outliers