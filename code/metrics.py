import networkx as nx
from networkx.algorithms import community

########### MODULARITY ###############
def compute_modularity_with_outliers(G, clusters, hubs, outliers):
    coms = [list(nodes) for nodes in clusters.values()]

    # 허브·이상치를 singleton clusters로 추가
    coms += [[u] for u in hubs]
    coms += [[u] for u in outliers]

    return community.modularity(G, coms, weight='weight')

def compute_modularity_clustered_only(G, clusters):
    clustered_nodes = set().union(*clusters.values())
    H = G.subgraph(clustered_nodes).copy()
    
    coms = [list(nodes) for nodes in clusters.values()]
    return community.modularity(H, coms, weight='weight')


########### CONDUCTANCE ###############
def conductance(G, cluster_nodes):
    S = set(cluster_nodes)
    cut_w = vol_S = vol_barS = 0.0
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        if (u in S) ^ (v in S):
            cut_w += w
        if u in S:
            vol_S += w
        else:
            vol_barS += w
        if v in S:
            vol_S += w
        else:
            vol_barS += w
    denom = min(vol_S, vol_barS)
    return cut_w/denom if denom>0 else 0.0


########### DENSITY ###############
def intra_density(G, cluster_nodes):
    S = set(cluster_nodes)
    total_w = sum(d['weight'] for u,v,d in G.subgraph(S).edges(data=True))
    n = len(S)
    max_edges = n*(n-1)/2
    return total_w/max_edges if max_edges>0 else 0.0

def inter_density(G, cluster_nodes):
    S = set(cluster_nodes)
    total_w = sum(d['weight']
                  for u,v,d in G.edges(data=True)
                  if (u in S) ^ (v in S))
    n = len(S)
    m = G.number_of_nodes() - n
    max_inter = n*m
    return total_w/max_inter if max_inter>0 else 0.0


########### ARI ###############

from sklearn.metrics import adjusted_rand_score

def compute_ARI(clusters, ground_truth):
    """
    Compute Adjusted Rand Index (ARI) given:
      - clusters: dict mapping cluster_id -> iterable of nodes
      - ground_truth: dict mapping node -> true_label (can be int or str)

    Returns:
      - ARI float in [-1,1]
    """
    # 1) Build predicted label mapping
    y_pred_map = {}
    for cid, members in clusters.items():
        for u in members:
            y_pred_map[u] = cid
    # 2) Prepare y_true, y_pred lists sorted by node ID
    nodes = sorted(ground_truth.keys())
    y_true = [ground_truth[u] for u in nodes]
    # assign any node not in a cluster to a special label (-1)
    y_pred = [y_pred_map.get(u, -1) for u in nodes]
    # 3) Compute ARI
    return adjusted_rand_score(y_true, y_pred)

# Example usage:
#
# clusters = {1: [1,2,3], 2: [4,5], 3: [6,7,8]}
# ground_truth = {1:'A',2:'A',3:'A',4:'B',5:'B',6:'C',7:'C',8:'C'}
# ari = compute_ari_from_dict(clusters, ground_truth)
# print(f"ARI = {ari:.4f}")


def compute_modularity(G, clusters):
    """
    Compute modularity for an undirected weighted graph G and given clusters.
    clusters: dict mapping cluster_id -> iterable of nodes
    Returns Q = sum over clusters of [E_C/m - (sum_deg_C/(2m))^2]
    """
    # Total edge weight (each undirected edge counted once)
    m = sum(data['weight'] for u, v, data in G.edges(data=True))
    Q = 0.0

    for cid, nodes in clusters.items():
        node_set = set(nodes)
        # Sum of weights of edges inside the cluster
        E_C = 0.0
        # Sum of degrees (strengths) of nodes in the cluster
        sum_deg_C = 0.0

        # Compute E_C
        for u in node_set:
            for v, data in G[u].items():
                if v in node_set:
                    E_C += data['weight']
        E_C /= 2.0  # each internal edge counted twice in the loop

        # Compute sum_deg_C
        for u in node_set:
            sum_deg_C += sum(data['weight'] for _, _, data in G.edges(u, data=True))

        # Community-level modularity contribution
        Q += (E_C / m) - (sum_deg_C / (2 * m))**2

    return Q