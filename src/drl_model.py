# drl_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv # GATv2 es a menudo más estable que GAT
from drl_features import NODE_FEATURE_DIM

class ActorCriticGAT(nn.Module):
    """
    Una red Actor-Critic que utiliza una Graph Attention Network (GAT) como codificador.
    """
    def __init__(self, input_dim=NODE_FEATURE_DIM, gat_hidden_dim=128, num_heads=4, output_mlp_dim=64):
        super(ActorCriticGAT, self).__init__()

        # --- Codificador GAT ---
        # Este procesa el grafo completo y genera embeddings ricos en contexto para cada nodo.
        self.gat1 = GATv2Conv(input_dim, gat_hidden_dim, heads=num_heads, concat=True, dropout=0.1, edge_dim=1)
        self.gat2 = GATv2Conv(gat_hidden_dim * num_heads, gat_hidden_dim, heads=1, concat=False, dropout=0.1, edge_dim=1)
        
        # --- Cabezas del Decodificador ---
        # El decodificador toma el embedding del nodo específico a decidir.

        # 1. Cabeza del Actor (Política)
        # Decide la probabilidad de asignar a la partición 0 o 1.
        self.policy_head = nn.Sequential(
            nn.Linear(gat_hidden_dim, output_mlp_dim),
            nn.ReLU(),
            nn.Linear(output_mlp_dim, 2) # 2 acciones: asignar a partición 0 o 1
        )

        # 2. Cabeza del Crítico (Valor)
        # Estima el retorno esperado (valor) desde el estado actual.
        self.value_head = nn.Sequential(
            nn.Linear(gat_hidden_dim, output_mlp_dim),
            nn.ReLU(),
            nn.Linear(output_mlp_dim, 1) # Un solo valor: la estimación del estado
        )

    def forward(self, x, edge_index, edge_weight, node_to_assign_idx):
        """
        Forward pass de la red.

        Args:
            x (torch.Tensor): Matriz de características de nodos.
            edge_index (torch.Tensor): Conectividad del grafo.
            edge_weight (torch.Tensor): Pesos de las aristas (usados por GATv2).
            node_to_assign_idx (int): El índice del nodo para el cual se debe tomar una decisión.
        
        Returns:
            tuple: Una tupla conteniendo:
                - action_logits (torch.Tensor): Logits para las acciones.
                - state_value (torch.Tensor): El valor estimado del estado.
        """
        # --- Codificador ---
        # Aplicar capas GAT
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.gat1(x, edge_index, edge_attr=edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        # En la última capa, los embeddings de las cabezas se promedian (concat=False)
        node_embeddings = self.gat2(x, edge_index, edge_attr=edge_weight)

        # --- Decodificador ---
        # Seleccionar el embedding del nodo específico para el que estamos decidiendo
        decision_node_embedding = node_embeddings[node_to_assign_idx]
        
        # Calcular la salida de ambas cabezas
        action_logits = self.policy_head(decision_node_embedding)
        state_value = self.value_head(decision_node_embedding)
        
        return action_logits, state_value