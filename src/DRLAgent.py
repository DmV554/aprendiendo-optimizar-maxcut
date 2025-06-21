# src/DRLAgent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_edge_index

class ActorCriticGAT(nn.Module):
    """
    Define la red Actor-Critic con una arquitectura GAT como encoder,
    como se describe en "Propuesta_DRL_para_MCP.pdf". 
    """
    def __init__(self, input_dim, hidden_dim, heads=4):
        """
        Inicializa la red.

        Args:
            input_dim (int): Dimensión de las características de cada nodo (en nuestro caso, 3).
            hidden_dim (int): Dimensión de la capa oculta de la GAT.
            heads (int): Número de cabezas de atención en la GAT. 
        """
        super(ActorCriticGAT, self).__init__()

        # Codificador GAT 
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)

        # Decodificador: Cabeza de Política (Actor) 
        self.policy_head = nn.Linear(hidden_dim, 2) # 2 acciones: partición 0 o 1

        # Decodificador: Cabeza de Valor (Crítico) 
        self.value_head = nn.Linear(hidden_dim, 1) # Estima el valor del estado

    def forward(self, state, current_node_idx=None):
        """
        Forward pass de la red.

        Args:
            state (tuple): El estado (adj_matrix, node_features).
            current_node_idx (int, optional): El índice del nodo a decidir. Requerido por el Actor.

        Returns:
            Si current_node_idx es not None -> (action_probs, state_value)
            Si current_node_idx es None -> (state_value)
        """
        adj_matrix, node_features = state
        
        # Convertir matriz de adyacencia densa a formato de lista de aristas para PyG
        edge_index, edge_attr = dense_to_edge_index(adj_matrix)
        
        # Codificador GAT
        x = F.relu(self.gat1(node_features, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        embeddings = self.gat2(x, edge_index, edge_attr=edge_attr) # Embeddings de todos los nodos 
        
        # Cabeza de Valor (Crítico) - Estima el valor del estado global
        # Usamos el embedding promedio del grafo como entrada para la cabeza de valor
        graph_embedding = embeddings.mean(dim=0)
        state_value = self.value_head(graph_embedding)

        if current_node_idx is not None:
            # Cabeza de Política (Actor) - Usa el embedding del nodo a decidir
            node_embedding = embeddings[current_node_idx]
            action_logits = self.policy_head(node_embedding)
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs, state_value
        else:
            # Si solo se necesita el valor (ej, al final del episodio)
            return state_value

class A2CAgent:
    """
    Implementa el agente que utiliza el algoritmo Advantage Actor-Critic (A2C).
    """
    def __init__(self, input_dim, hidden_dim, lr=1e-4, gamma=0.99, entropy_coef=0.01):
        """
        Inicializa el agente.

        Args:
            input_dim (int): Dimensión de las características de entrada de la red.
            hidden_dim (int): Dimensión de la capa oculta de la red.
            lr (float): Tasa de aprendizaje.
            gamma (float): Factor de descuento para recompensas futuras.
            entropy_coef (float): Coeficiente para la regularización por entropía. 
        """
        self.policy_network = ActorCriticGAT(input_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # Buffers para el episodio actual
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = [] # AÑADIDO: Buffer para almacenar la entropía de cada paso

    def select_action(self, state, current_node_idx):
        """
        Selecciona una acción basada en la política actual (muestreo estocástico).
        
        Args:
            state (tuple): El estado actual del entorno.
            current_node_idx (int): El nodo para el cual se debe tomar una decisión.

        Returns:
            int: La acción seleccionada (0 o 1).
        """
        action_probs, state_value = self.policy_network(state, current_node_idx)
        
        # Crear una distribución categórica y muestrear una acción
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Guardar el log de la probabilidad de la acción y el valor del estado
        self.log_probs.append(dist.log_prob(action))
        self.values.append(state_value)
        self.entropies.append(dist.entropy()) # AÑADIDO: Guardar la entropía de la distribución
        
        return action.item()

    def store_outcome(self, reward, done):
        """Almacena la recompensa y el flag de 'done' del último paso."""
        self.rewards.append(reward)
        self.dones.append(done)

    def update_policy(self, last_state):
        """
        Actualiza los pesos de la red al final de un episodio usando el marco A2C.
        """
        # Calcular el valor del estado terminal
        with torch.no_grad():
            last_value = self.policy_network(last_state) if not self.dones[-1] else torch.tensor(0.0)
            self.values.append(last_value)

        # Calcular los retornos y ventajas 
        returns = []
        R = last_value
        for t in reversed(range(len(self.rewards))):
            R = self.rewards[t] + self.gamma * R * (1 - self.dones[t])
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        values = torch.cat(self.values).squeeze()
        
        # Normalizar retornos puede ayudar a estabilizar el entrenamiento
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        advantages = returns - values
        
        # Calcular la pérdida del Actor (política) y Crítico (valor)
        log_probs = torch.stack(self.log_probs)
        
        policy_loss = (-log_probs * advantages.detach()).mean() # 
        value_loss = F.mse_loss(returns, values) #

        # Queremos maximizar la entropía, por lo que minimizamos su negativo.
        # Calculamos la media de la entropía a lo largo del episodio.
        entropy_loss = torch.stack(self.entropies).mean() 
        
        
        # PÉRDIDA TOTAL ACTUALIZADA
        # Restamos la pérdida de entropía multiplicada por su coeficiente.
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy_loss
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Limpiar buffers para el próximo episodio
        self.clear_buffers()
        
        return loss.item()

    def clear_buffers(self):
        """Limpia las memorias del episodio."""
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.dones[:]
        del self.entropies[:] # AÑADIDO: Limpiar el buffer de entropías
