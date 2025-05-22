from collections import Counter

input_path = 'CollegeMsg.txt'
output_path = 'collegemsg_edges.dat'

counter = Counter()

with open(input_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        u, v, _ = parts
        # u,v 와 v,u 를 합치기 위해 항상 작은 ID를 앞에
        a, b = sorted((int(u), int(v)))
        counter[(a, b)] += 1

# 2) 무향 가중치 엣지 리스트로 저장
with open(output_path, 'w') as out:
    for (u, v), w in counter.items():
        out.write(f"{u} {v} {w}\n")