import networkx as nx
import numpy as np

def prune_by_threshold(G, thresh):
    G2 = G.copy()
    for u,v,data in list(G.edges(data=True)):
        if data['weight'] <= thresh:
            G2.remove_edge(u,v)
    return G2

def prune_by_percentile(G, q=50):
    # weight 값을 배열로
    ws = np.array([d['weight'] for _,_,d in G.edges(data=True)])
    thresh = np.percentile(ws, q)
    return prune_by_threshold(G, thresh)

# # 1) 내장 그래프 로드
G = nx.les_miserables_graph()
print(len(G.nodes))
print(len(G.edges))

# # 2) 'value' 필드를 'weight'로 복사
# for u, v, data in G.edges(data=True):
#     data['weight'] = data.get('weight', 1)

# # 3) 정수 ID로 노드 리레이블 (선택)
# mapping = {name: idx for idx, name in enumerate(G.nodes(), start=1)}
# G = nx.relabel_nodes(G, mapping)

# G2 = G.copy()
# for u,v,data in list(G.edges(data=True)):
#     if data['weight'] < 2:
#         G2.remove_edge(u,v)

# # 4) network.dat로 저장
# with open("real/les_miserable_edges.dat", "w") as f:
#     for u, v, data in G2.edges(data=True):
#         w = data.get("weight", 1)
#         f.write(f"{u} {v} {w}\n")

# print("✅ network.dat 파일이 생성되었습니다.")

# G = nx.read_weighted_edgelist("butterfly_edges.dat", nodetype=int)

# # 2) weight == 1인 간선 제거
# # G = prune_by_threshold(G, 1)
# G = prune_by_percentile(G)

# # 3) 결과 저장
# nx.write_weighted_edgelist(G, "real/butterfly_edges.dat", delimiter=' ')

# print("✅ pruned graph saved to collegemsg_pruned.dat")