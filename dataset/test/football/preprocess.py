# preprocess.py
import argparse
import networkx as nx
from collections import defaultdict

def read_gml_undirected(path):
    G = nx.read_gml(path)
    if G.is_directed():
        G = G.to_undirected()
    return G

def build_index_mapping(G, start_index=1):
    # GML에서 노드는 문자열(팀명) 키일 수 있으니, 1-based 정수로 매핑
    node_to_idx = {node: i+start_index for i, node in enumerate(G.nodes())}
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    return node_to_idx, idx_to_node

def write_labels(G, node_to_idx, out_path):
    # GT = node의 'value' (컨퍼런스 번호)
    with open(out_path, "w", encoding="utf-8") as f:
        for node, data in G.nodes(data=True):
            label = data.get("value", -1)
            f.write(f"{node_to_idx[node]} {label}\n")

def write_mapping(idx_to_node, G, out_path):
    # 숫자ID ↔ 팀명 ↔ 컨퍼런스 기록
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("#id\tteam\tconference_value\n")
        for idx in sorted(idx_to_node):
            name = idx_to_node[idx]
            conf = G.nodes[name].get("value", -1)
            f.write(f"{idx}\t{name}\t{conf}\n")

def write_dat_constant_weight(G, node_to_idx, out_path, w=1):
    with open(out_path, "w", encoding="utf-8") as f:
        for u, v in G.edges():
            f.write(f"{node_to_idx[u]} {node_to_idx[v]} {w}\n")

def write_dat_common_neighbors_plus_one(G, node_to_idx, out_path):
    # 공통 이웃 수 + 1
    # 이웃 set을 미리 구성
    neigh = {u: set(G.neighbors(u)) for u in G.nodes()}
    with open(out_path, "w", encoding="utf-8") as f:
        for u, v in G.edges():
            w = len(neigh[u].intersection(neigh[v])) + 1
            f.write(f"{node_to_idx[u]} {node_to_idx[v]} {w}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gml", type=str, default="football.gml", help="football.gml 경로")
    ap.add_argument("--out_prefix", type=str, default="football", help="출력 파일 접두어")
    ap.add_argument("--start_index", type=int, default=1, help="노드 인덱스 시작(1 권장)")
    args = ap.parse_args()

    G = read_gml_undirected(args.gml)
    node_to_idx, idx_to_node = build_index_mapping(G, start_index=args.start_index)

    # 1) baseline: weight=1
    write_dat_constant_weight(G, node_to_idx, f"{args.out_prefix}_w1.dat", w=1)

    # 2) 구조 기반: 공통이웃+1
    write_dat_common_neighbors_plus_one(G, node_to_idx, f"{args.out_prefix}_commonOpp.dat")

    # GT 라벨과 매핑
    write_labels(G, node_to_idx, f"{args.out_prefix}.labels")
    write_mapping(idx_to_node, G, f"{args.out_prefix}.map")

    print("Saved:")
    print(f"  {args.out_prefix}_w1.dat                (모든 엣지 weight=1)")
    print(f"  {args.out_prefix}_commonOpp.dat        (공통이웃 수 + 1 가중)")
    print(f"  {args.out_prefix}.labels               (node  label[conference])")
    print(f"  {args.out_prefix}.map                  (#id  team  conference)")

if __name__ == "__main__":
    main()
