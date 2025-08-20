import networkx as nx
import numpy as np
import random
from collections import defaultdict

def generate_weighted_lfr_datasets():
    """
    여러 전략으로 가중치 LFR 그래프 데이터셋 생성 및 .dat 형식으로 저장
    """
    
    # 다양한 파라미터 설정
    configs = [
        {
            'name': 'community_biased_v1',
            'n': 500,
            'tau1': 2.5,
            'tau2': 1.5,
            'mu': 0.25,
            'min_degree': 10,
            'max_degree': 50,
            'strategy': 'community_biased'
        },
        {
            'name': 'community_biased_v2',
            'n': 500,
            'tau1': 2.0,
            'tau2': 1.5,
            'mu': 0.3,
            'min_degree': 15,
            'max_degree': 60,
            'strategy': 'community_biased_extreme'
        },
        {
            'name': 'hub_centric_v1',
            'n': 500,
            'tau1': 2.5,
            'tau2': 1.5,
            'mu': 0.2,
            'min_degree': 10,
            'max_degree': 80,
            'strategy': 'hub_centric'
        },
        {
            'name': 'power_law_v1',
            'n': 500,
            'tau1': 2.3,
            'tau2': 1.3,
            'mu': 0.25,
            'min_degree': 12,
            'max_degree': 70,
            'strategy': 'power_law'
        },
        {
            'name': 'mixed_weights_v1',
            'n': 500,
            'tau1': 2.5,
            'tau2': 1.5,
            'mu': 0.35,
            'min_degree': 10,
            'max_degree': 50,
            'strategy': 'mixed'
        }
    ]
    
    for config in configs:
        print(f"Generating dataset: {config['name']}...")
        
        # LFR 그래프 생성
        G = nx.LFR_benchmark_graph(
            n=config['n'],
            tau1=config['tau1'],
            tau2=config['tau2'],
            mu=config['mu'],
            min_degree=config['min_degree'],
            max_degree=config['max_degree'],
            min_community=20,
            max_community=100,
            seed=42
        )
        
        # 커뮤니티 정보 추출
        communities = defaultdict(list)
        for node in G.nodes():
            for comm in G.nodes[node]['community']:
                communities[comm].append(node)
        
        # 가중치 할당 전략
        strategy = config['strategy']
        
        if strategy == 'community_biased':
            # 커뮤니티 내부: 불균형 분포, 외부: 약한 연결
            for u, v in G.edges():
                u_comms = set(G.nodes[u]['community'])
                v_comms = set(G.nodes[v]['community'])
                
                if u_comms & v_comms:  # 같은 커뮤니티
                    if random.random() < 0.15:  # 15%는 매우 강한 연결
                        weight = np.random.uniform(8.0, 10.0)
                    elif random.random() < 0.4:  # 25%는 강한 연결
                        weight = np.random.uniform(4.0, 7.0)
                    else:  # 60%는 중간 연결
                        weight = np.random.uniform(1.0, 3.0)
                else:  # 다른 커뮤니티
                    if random.random() < 0.03:  # 3%는 노이즈
                        weight = np.random.uniform(3.0, 5.0)
                    else:
                        weight = np.random.uniform(0.1, 0.8)
                
                G[u][v]['weight'] = weight
        
        elif strategy == 'community_biased_extreme':
            # 더 극단적인 커뮤니티 기반 불균형
            for u, v in G.edges():
                u_comms = set(G.nodes[u]['community'])
                v_comms = set(G.nodes[v]['community'])
                
                if u_comms & v_comms:
                    # 멱법칙 분포 사용
                    if random.random() < 0.1:  # 10%는 초강력 연결
                        weight = np.random.uniform(15.0, 20.0)
                    elif random.random() < 0.2:
                        weight = np.random.uniform(5.0, 10.0)
                    else:
                        weight = np.random.pareto(2.0) + 0.5
                        weight = min(weight, 5.0)  # 상한 설정
                else:
                    weight = np.random.exponential(0.3)
                    weight = min(weight, 2.0)
                
                G[u][v]['weight'] = weight
        
        elif strategy == 'hub_centric':
            # 허브 노드 중심의 불균형
            degrees = dict(G.degree())
            degree_threshold_high = np.percentile(list(degrees.values()), 85)
            degree_threshold_low = np.percentile(list(degrees.values()), 30)
            
            for u, v in G.edges():
                u_is_hub = degrees[u] > degree_threshold_high
                v_is_hub = degrees[v] > degree_threshold_high
                u_is_leaf = degrees[u] < degree_threshold_low
                v_is_leaf = degrees[v] < degree_threshold_low
                
                if u_is_hub and v_is_hub:
                    # 허브 간 연결은 매우 강함
                    weight = np.random.uniform(10.0, 15.0)
                elif u_is_hub or v_is_hub:
                    # 허브와의 연결은 극단적 불균형
                    if random.random() < 0.25:
                        weight = np.random.uniform(6.0, 12.0)
                    else:
                        weight = np.random.uniform(0.2, 1.5)
                elif u_is_leaf and v_is_leaf:
                    # 리프 노드 간은 약함
                    weight = np.random.uniform(0.1, 1.0)
                else:
                    # 일반 노드 간
                    weight = np.random.uniform(1.0, 4.0)
                
                G[u][v]['weight'] = weight
        
        elif strategy == 'power_law':
            # 멱법칙 분포 (극단적 불균형)
            weights = np.random.pareto(a=1.2, size=G.number_of_edges()) + 0.01
            weights = weights * 15 / weights.max()  # 0.01 ~ 15 범위
            
            for i, (u, v) in enumerate(G.edges()):
                G[u][v]['weight'] = weights[i]
        
        elif strategy == 'mixed':
            # 혼합 전략: 커뮤니티 + 허브 + 노이즈
            degrees = dict(G.degree())
            degree_threshold = np.percentile(list(degrees.values()), 80)
            
            for u, v in G.edges():
                u_comms = set(G.nodes[u]['community'])
                v_comms = set(G.nodes[v]['community'])
                u_is_hub = degrees[u] > degree_threshold
                v_is_hub = degrees[v] > degree_threshold
                
                if (u_comms & v_comms) and (u_is_hub or v_is_hub):
                    # 커뮤니티 내 허브 연결: 최강
                    weight = np.random.uniform(12.0, 18.0)
                elif u_comms & v_comms:
                    # 일반 커뮤니티 내부
                    if random.random() < 0.3:
                        weight = np.random.uniform(4.0, 8.0)
                    else:
                        weight = np.random.gamma(2, 1)
                        weight = min(weight, 6.0)
                elif u_is_hub or v_is_hub:
                    # 커뮤니티 외부 허브 연결
                    weight = np.random.exponential(0.5)
                    weight = min(weight, 3.0)
                else:
                    # 일반 외부 연결
                    weight = np.random.uniform(0.05, 0.5)
                
                # 가끔 노이즈 추가
                if random.random() < 0.02:
                    weight *= np.random.uniform(5, 10)
                
                G[u][v]['weight'] = weight
        
        # 데이터 파일 저장 (.dat 형식)
        data_filename = f"{config['name']}_data.dat"
        with open(data_filename, 'w') as f:
            # 엣지 정보 저장 (node1 node2 weight)
            for u, v, data in G.edges(data=True):
                f.write(f"{u} {v} {data['weight']:.6f}\n")
        
        print(f"  - Data saved to {data_filename}")
        
        # 레이블(커뮤니티) 파일 저장 (.dat 형식)
        labels_filename = f"{config['name']}_labels.dat"
        with open(labels_filename, 'w') as f:
            # 노드별 커뮤니티 레이블 저장
            # 노드가 여러 커뮤니티에 속할 수 있으므로 첫 번째 커뮤니티를 주 레이블로 사용
            for node in sorted(G.nodes()):
                comm_list = list(G.nodes[node]['community'])
                # 첫 번째 커뮤니티를 대표 레이블로 사용
                primary_label = comm_list[0] if comm_list else -1
                f.write(f"{node} {primary_label}\n")
        
        print(f"  - Labels saved to {labels_filename}")
        
        # 통계 정보 출력
        print(f"  - Statistics: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
              f"communities={len(communities)}")
        print()

if __name__ == "__main__":
    
    # 데이터셋 생성
    generate_weighted_lfr_datasets()