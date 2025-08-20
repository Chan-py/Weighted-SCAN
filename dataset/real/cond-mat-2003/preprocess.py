import networkx as nx

# GML 파일 불러오기
G = nx.read_gml("./cond-mat-2003.gml", label="id")  # label="id" 하면 node id 기반으로 읽음

# network.dat 형식으로 저장
with open("network.dat", "w") as f:
    for u, v, data in G.edges(data=True):
        weight = data.get("value", 1)  # weight 없으면 1로 설정
        f.write(f"{u} {v} {weight}\n")
