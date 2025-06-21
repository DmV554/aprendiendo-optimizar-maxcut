# src/DRLEnvironment.py

import torch
import numpy as np
from MAXCUT import MaxCutInstance, MaxCutState

class DRLEnvironment:
    """
    Define el entorno para el agente DRL para resolver Max-Cut de forma constructiva.
    Se basa en la formulación del PDF "Propuesta_DRL_para_MCP.pdf".
    """
    def __init__(self, instance: MaxCutInstance):
        """
        Inicializa el entorno con una instancia específica de Max-Cut.

        Args:
            instance (MaxCutInstance): La instancia del problema a resolver.
        """
        self.instance = instance
        self.num_nodes = self.instance.num_nodes
        self.weights_matrix = torch.from_numpy(self.instance.weights_matrix).float()
        self.node_order = None
        self.current_step = 0
        self.current_partition = None

    def reset(self, shuffle_nodes=True):
        """
        Reinicia el entorno para un nuevo episodio.

        Args:
            shuffle_nodes (bool): Si es True, el orden en que se visitan los nodos será aleatorio.

        Returns:
            tuple: El estado inicial (matriz de adyacencia, matriz de características de nodos).
        """
        self.current_step = 0
        # La partición se inicializa con -1 (no asignado)
        self.current_partition = torch.full((self.num_nodes,), -1, dtype=torch.float)

        if shuffle_nodes:
            self.node_order = torch.randperm(self.num_nodes)
        else:
            self.node_order = torch.arange(self.num_nodes)

        return self._get_state()

    def _get_state(self):
        """
        Construye la representación del estado actual.

        Returns:
            tuple: (adjacency_matrix, node_features_matrix)
        """
        # Componente Dinámico: Matriz de características de nodos X_t
        # 
        # Feature 1: Indicador de Asignación (1 si está asignado, 0 si no). 
        # Feature 2: Identificador de Partición (0 o 1 si está asignado, -1 si no). 
        # Se adapta ligeramente la codificación para que sea más fácil para la GNN.
        # Usaremos one-hot encoding para el estado de asignación.
        # [No Asignado, Asignado a P0, Asignado a P1]
        node_features = torch.zeros(self.num_nodes, 3)
        for i in range(self.num_nodes):
            if self.current_partition[i] == -1:
                node_features[i, 0] = 1 # No asignado
            elif self.current_partition[i] == 0:
                node_features[i, 1] = 1 # Asignado a P0
            else: # == 1
                node_features[i, 2] = 1 # Asignado a P1

        # Componente Estático: Matriz de adyacencia G 
        # El estado completo es la tupla (G, X_t) 
        return (self.weights_matrix, node_features)

    def step(self, action: int):
        """
        Avanza un paso en el entorno aplicando una acción.

        Args:
            action (int): La acción a tomar (0 para partición 0, 1 para partición 1).

        Returns:
            tuple: (next_state, reward, done)
        """
        if self.is_done():
            raise Exception("El episodio ha terminado. Llama a reset() para empezar uno nuevo.")

        # Obtener el índice del nodo a decidir en este paso
        node_to_assign_idx = self.node_order[self.current_step].item()

        # Asignar el nodo a la partición según la acción
        self.current_partition[node_to_assign_idx] = action

        # Calcular la recompensa densa e inmediata 
        reward = self._calculate_reward(node_to_assign_idx, action)

        # Avanzar al siguiente paso
        self.current_step += 1
        done = self.is_done()

        next_state = self._get_state()

        return next_state, reward, done

    def _calculate_reward(self, assigned_node_idx: int, assigned_partition: int):
        """
        Calcula la recompensa por asignar un nodo a una partición.
        La recompensa es la suma de los pesos de las aristas que conectan
        el nodo recién asignado con los nodos previamente asignados en la partición opuesta.
        
        """
        reward = 0.0
        # Nodos ya asignados (excluyendo el actual)
        assigned_mask = (self.current_partition != -1) & (torch.arange(self.num_nodes) != assigned_node_idx)
        
        # Nodos en la partición opuesta
        opposite_partition_mask = self.current_partition != assigned_partition
        
        # Nodos vecinos que cumplen ambas condiciones
        neighbors_in_opposite_cut = assigned_mask & opposite_partition_mask

        # Sumar los pesos de las aristas
        if neighbors_in_opposite_cut.any():
            reward = self.weights_matrix[assigned_node_idx, neighbors_in_opposite_cut].sum().item()

        return reward

    def is_done(self):
        """
        Verifica si el episodio ha terminado (todos los nodos han sido asignados).
        
        """
        return self.current_step >= self.num_nodes

    def get_final_cut_value(self):
        """
        Calcula el valor del corte de la solución final construida.
        """
        if not self.is_done():
            return 0.0
            
        cut_value = 0.0
        partition_np = self.current_partition.numpy()
        # Usamos la evaluación de tu clase MaxCutInstance para consistencia
        cut_value = self.instance.evaluate_partition(partition_np.tolist())
        return cut_value