import numpy as np
import networkx as nx

from utils import eps_grid_by_quantiles
import similarity

network = "collegemsg"
output_file="commands.txt"
dataset = "../dataset/real/" + network + "/network.dat"
G = nx.read_weighted_edgelist(dataset, nodetype=int)
    
similarities = {"scan" : similarity.scan_similarity, "wscan" : similarity.wscan_similarity,
       "cosine" : similarity.cosine_similarity, "Gen" : similarity.Gen_wscan_similarity, 
       "Jaccard" : similarity.weighted_jaccard_similarity}
commands = []

for sim in similarities:
    similarity_func = similarities[sim]
    # eps 후보 설정
    if sim == "wscan":
        # wscan은 일반적으로 더 큰 값 사용
        eps_candidates = eps_grid_by_quantiles(G, similarity_func)
    else:
        eps_candidates = np.arange(0.1, 0.95, 0.05)
    
    # mu 범위: 2, 3, 4, 5
    mu_range = range(2, 6)

    if sim == "Gen":
        gamma_range = np.arange(0.5, 1.05, 0.1)
    else:
        gamma_range = range(1, 2)
    
    # 모든 조합에 대해 명령어 생성
    for eps in eps_candidates:
        for mu in mu_range:
            for gamma in gamma_range:
                command = f"python main.py --network {network} --eps {eps:.2f} --mu {mu} --gamma {gamma:.1f} --similarity {sim}"
                commands.append(command)

# txt 파일로 저장
with open(output_file, 'w') as f:
    for i, command in enumerate(commands):
        # 마지막 명령어가 아니면 & 추가
        if i < len(commands) - 1:
            f.write(command + "&\n")
        else:
            f.write(command + "\n")

print(f"Generated {len(commands)} commands in {output_file}")

print(f"\n현재 명령어 수: {len(commands)}개")