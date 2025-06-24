# drl_features.py

import numpy as np
import torch
# Asumimos que la clase MaxCutState está definida en MAXCUT.py
from MAXCUT import MaxCutState 

# La dimensión de las características de cada nodo. Es crucial que coincida con el modelo.
# [is_unassigned, is_in_part_0, is_in_part_1, degree, sum_weights_to_part_0, sum_weights_to_part_1]
NODE_FEATURE_DIM = 8

def calculate_dynamic_features(v_idx, current_partition, weights_matrix):
    sum_to_part_0 = 0.0
    sum_to_part_1 = 0.0

    for j_idx, assignment in enumerate(current_partition):
        if v_idx == j_idx or assignment == -1:
            continue
        weight = weights_matrix[v_idx, j_idx]
        if weight == 0: continue
        if assignment == 0: sum_to_part_0 += weight
        elif assignment == 1: sum_to_part_1 += weight
            
    # La ganancia si v_idx va a la partición 0 es la suma de pesos a la partición 1
    greedy_gain_if_0 = sum_to_part_1
    # La ganancia si v_idx va a la partición 1 es la suma de pesos a la partición 0
    greedy_gain_if_1 = sum_to_part_0
            
    return sum_to_part_0, sum_to_part_1, greedy_gain_if_0, greedy_gain_if_1

def state_to_graph(state: MaxCutState):
    instance = state.inst_info
    num_nodes = instance.num_nodes
    weights_matrix = instance.weights_matrix
    partition = state.partition

    degrees = np.sum(weights_matrix != 0, axis=1)
    node_features = np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)

    for i in range(num_nodes):
        assign_state = partition[i]
        is_unassigned = 1.0 if assign_state == -1 else 0.0
        is_in_part_0 = 1.0 if assign_state == 0 else 0.0
        is_in_part_1 = 1.0 if assign_state == 1 else 0.0
        
        sum_to_p0, sum_to_p1, gain0, gain1 = calculate_dynamic_features(i, partition, weights_matrix)

        node_features[i, :] = [
            is_unassigned, is_in_part_0, is_in_part_1,
            degrees[i], sum_to_p0, sum_to_p1,
            gain0, gain1 # <-- NUEVOS FEATURES
        ]

    adj_coo = torch.from_numpy(weights_matrix).to_sparse_coo()
    edge_index = adj_coo.indices().long()
    edge_weight = adj_coo.values().float()
    
    return torch.tensor(node_features, dtype=torch.float32), edge_index, edge_weight