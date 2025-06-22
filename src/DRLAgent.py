# src/DRLAgent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class ActorCriticGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=4):
        super(ActorCriticGAT, self).__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.1)

        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1)

        self.gat3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.1)

        self.policy_head = nn.Linear(hidden_dim * 2, 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, data, current_node_idx: int = None):
        """
        Forward pass de la red.

        Args:
            data: El estado del grafo. Puede ser un objeto Data de PyG o una tupla (x, edge_index, edge_attr).
            current_node_idx (int, optional): El índice del nodo a decidir. Requerido por el Actor.

        Returns:
            Si current_node_idx is not None -> (action_probs, state_value)
            Si current_node_idx es None -> (state_value)
        """
        # Manejo flexible del input - puede ser Data object o tupla
        if isinstance(data, Data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        elif isinstance(data, tuple) and len(data) == 3:
            x, edge_index, edge_attr = data
        else:
            raise ValueError(f"Formato de data no soportado: {type(data)}. Se esperaba Data object o tupla (x, edge_index, edge_attr)")
        
        # Verificar que los tensores tienen las dimensiones correctas
        if x is None or edge_index is None:
            raise ValueError("x y edge_index no pueden ser None")
            
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.gat2(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)

        embeddings = self.gat3(x, edge_index, edge_attr=edge_attr)
        graph_embedding = embeddings.mean(dim=0)
        state_value = self.value_head(graph_embedding)

        if current_node_idx is not None:
            node_embedding = embeddings[current_node_idx]

            combined_embedding = torch.cat([node_embedding, graph_embedding])

            action_logits = self.policy_head(combined_embedding)
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs, state_value
        else:
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

    def select_action(self, state, current_node_idx: int):
        """
        Selecciona una acción basada en la política actual (muestreo estocástico). 
        
        Args:
            state: El estado actual del entorno (Data object o tupla).
            current_node_idx (int): El nodo para el cual se debe tomar una decisión.

        Returns:
            int: La acción seleccionada (0 o 1).
        """
        # El estado ahora debería ser un objeto Data de PyTorch Geometric
        
        # Mantener el modelo en modo de entrenamiento para preservar gradientes
        self.policy_network.train()
        
        # NO usar torch.no_grad() aquí porque necesitamos los gradientes
        action_probs, state_value = self.policy_network(state, current_node_idx)

        # Crear una distribución categórica y muestrear una acción para balancear exploración/explotación.
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Guardar datos relevantes para la actualización de la política.
        # Estos tensores deben mantener sus gradientes
        self.log_probs.append(dist.log_prob(action))
        self.values.append(state_value)
        self.entropies.append(dist.entropy())
        
        return action.item()

    def store_outcome(self, reward, done):
        """Almacena la recompensa y el flag de 'done' del último paso."""
        self.rewards.append(reward)
        self.dones.append(done)

    def update_policy(self, last_state):
        """
        Actualiza los pesos de la red al final de un episodio usando el marco A2C.
        """
        # Verificar que tenemos experiencias para entrenar
        if len(self.rewards) == 0:
            print("Warning: No hay experiencias para entrenar")
            return 0.0
        
        # Calcular el valor del estado terminal para el bootstrapping.
        with torch.no_grad():
            # Si el episodio no ha terminado, obtenemos el valor del último estado.
            # Si terminó, el valor es 0.
            if not self.dones[-1]:
                last_value = self.policy_network(last_state)
            else:
                last_value = torch.tensor(0.0)
        
        # Calcular los retornos y ventajas de forma retrospectiva.
        returns = []
        R = last_value
        for t in reversed(range(len(self.rewards))):
            R = self.rewards[t] + self.gamma * R * (1 - self.dones[t])
            returns.insert(0, R)
            
        returns = torch.tensor(returns, requires_grad=False)
        
        # Verificar que tenemos valores para concatenar
        if len(self.values) == 0:
            print("Warning: No hay valores para concatenar")
            return 0.0
            
        values = torch.cat(self.values).squeeze()
        
        # Asegurar que values requiere gradientes
        if not values.requires_grad:
            print("Warning: Los valores no requieren gradientes")
            return 0.0
        
        # Normalizar retornos o ventajas puede estabilizar el entrenamiento.
        advantages = returns - values.detach()  # Detach values para calcular advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calcular las pérdidas del Actor y del Crítico.
        log_probs = torch.stack(self.log_probs)
        
        # Verificar que log_probs requiere gradientes
        if not log_probs.requires_grad:
            print("Warning: Los log_probs no requieren gradientes")
            return 0.0
        
        # Pérdida del Actor (Política): anima a tomar acciones con alta ventaja.
        policy_loss = (-log_probs * advantages.detach()).mean()
        
        # Pérdida del Crítico (Valor): MSE entre los retornos reales y los predichos.
        value_loss = F.mse_loss(values, returns)

        # Pérdida de Entropía: Anima a la política a ser más estocástica para mejorar la exploración. 
        entropy_loss = torch.stack(self.entropies).mean()
        
        # Pérdida total combinada.
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy_loss
        
        # Verificar que la pérdida requiere gradientes
        if not loss.requires_grad:
            print("Warning: La pérdida no requiere gradientes")
            print(f"policy_loss.requires_grad: {policy_loss.requires_grad}")
            print(f"value_loss.requires_grad: {value_loss.requires_grad}")
            print(f"entropy_loss.requires_grad: {entropy_loss.requires_grad}")
            return 0.0
        
        # Optimización.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Limpiar buffers para el próximo episodio.
        self.clear_buffers()
        
        return loss.item()
    
    def select_action_greedy(self, state, current_node_idx: int):
        """
        Selecciona la mejor acción posible de forma determinista (greedy) para la evaluación.
        """
        self.policy_network.eval() # Poner la red en modo de evaluación
        with torch.no_grad(): # Desactivar el cálculo de gradientes para la inferencia
            action_probs, _  = self.policy_network(state, current_node_idx)
            action = torch.argmax(action_probs).item() # Elegir la acción con la probabilidad más alta
        return action

    def clear_buffers(self):
        """Limpia las memorias del episodio."""
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.entropies.clear()