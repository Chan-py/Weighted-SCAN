# Spring layout + matplotlib 예시
import matplotlib.pyplot as plt
import networkx as nx

def plot_clusters(G, clusters):
    pos = nx.spring_layout(G, weight='weight')
    cmap = plt.get_cmap('tab20')
    plt.figure(figsize=(8,8))
    for cid, nodes in clusters.items():
        nx.draw_networkx_nodes(G, pos,
                               nodelist=list(nodes),
                               node_size=50,
                               node_color=[cmap(cid % 20)])
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.axis('off')
    plt.show()

def load_ground_truth(labels_path):
    """label.dat 포맷: node true_label (공백 구분)"""
    gt = {}
    with open(labels_path, 'r') as f:
        for line in f:
            node, label = line.strip().split()
            gt[int(node)] = label  # label이 int면 int(label)
    return gt