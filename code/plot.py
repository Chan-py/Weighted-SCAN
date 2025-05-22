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