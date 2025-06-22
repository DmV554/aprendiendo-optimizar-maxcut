# drl_features.py

import numpy as np
import torch
# Asumimos que la clase MaxCutState está definida en MAXCUT.py
from MAXCUT import MaxCutState 

# La dimensión de las características de cada nodo. Es crucial que coincida con el modelo.
# [is_unassigned, is_in_part_0, is_in_part_1, degree, sum_weights_to_part_0, sum_weights_to_part_1]
NODE_FEATURE_DIM = 6

def calculate_dynamic_features(v_idx, current_partition, weights_matrix):
    """
    Calcula características dinámicas para un nodo v_idx basadas en la partición parcial.
    Adaptado de tu implementación anterior.
    """
    sum_to_part_0 = 0.0
    sum_to_part_1 = 0.0

    for j_idx, assignment in enumerate(current_partition):
        if v_idx == j_idx or assignment == -1:
            continue
        
        weight = weights_matrix[v_idx, j_idx]
        if weight == 0:
            continue

        if assignment == 0:
            sum_to_part_0 += weight
        elif assignment == 1:
            sum_to_part_1 += weight
            
    return sum_to_part_0, sum_to_part_1

def state_to_graph(state: MaxCutState):
    """
    Convierte un MaxCutState en una representación de grafo para PyTorch Geometric.

    Args:
        state (MaxCutState): El estado actual del entorno.

    Returns:
        tuple: Una tupla conteniendo:
            - x (torch.Tensor): Matriz de características de nodos.
            - edge_index (torch.Tensor): Conectividad del grafo en formato COO.
            - edge_weight (torch.Tensor): Pesos de las aristas.
    """
    instance = state.inst_info
    num_nodes = instance.num_nodes
    weights_matrix = instance.weights_matrix
    partition = state.partition

    # 1. Calcular características estáticas una vez
    degrees = np.sum(weights_matrix != 0, axis=1)

    # 2. Construir la matriz de características de nodos (x)
    node_features = np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)

    for i in range(num_nodes):
        # Features de asignación (One-Hot Encoding)
        assign_state = partition[i]
        is_unassigned = 1.0 if assign_state == -1 else 0.0
        is_in_part_0 = 1.0 if assign_state == 0 else 0.0
        is_in_part_1 = 1.0 if assign_state == 1 else 0.0
        
        # Features dinámicas
        sum_to_p0, sum_to_p1 = calculate_dynamic_features(i, partition, weights_matrix)

        node_features[i, :] = [
            is_unassigned,
            is_in_part_0,
            is_in_part_1,
            degrees[i],
            sum_to_p0,
            sum_to_p1
        ]

    # 3. Construir la estructura de aristas (edge_index) y pesos (edge_weight)
    adj_coo = torch.from_numpy(weights_matrix).to_sparse_coo()
    edge_index = adj_coo.indices().long()
    edge_weight = adj_coo.values().float()
    
    return torch.tensor(node_features, dtype=torch.float32), edge_index, edge_weight