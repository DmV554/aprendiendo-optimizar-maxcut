# src/DRLEnvironment.py

import torch
import numpy as np
from torch_geometric.data import Data
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
        self.static_degrees = None
        self.static_weighted_degrees = None
        
        # Precompute edge_index and edge_attr from the weights matrix
        self.edge_index, self.edge_attr = self._create_edge_tensors()

    def _create_edge_tensors(self):
        """
        Crea los tensores edge_index y edge_attr a partir de la matriz de pesos.
        
        Returns:
            tuple: (edge_index, edge_attr) donde edge_index es de forma [2, num_edges] 
                   y edge_attr contiene los pesos de las aristas.
        """
        # Encontrar todas las aristas (donde weight > 0)
        edges = []
        edge_weights = []
        
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):  # Solo considerar aristas únicas (i < j)
                if self.weights_matrix[i, j] > 0:
                    # Agregar arista en ambas direcciones para grafo no dirigido
                    edges.extend([[i, j], [j, i]])
                    edge_weights.extend([self.weights_matrix[i, j], self.weights_matrix[i, j]])
        
        if len(edges) == 0:
            # Grafo sin aristas
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_attr

    def reset(self, ordering_strategy='random'):
        """
        Reinicia el entorno para un nuevo episodio.

        Args:
            shuffle_nodes (bool): Si es True, el orden en que se visitan los nodos será aleatorio.

        Returns:
            Data: El estado inicial como objeto Data de PyTorch Geometric.
        """
        self.current_step = 0
        # La partición se inicializa con -1 (no asignado)
        self.current_partition = torch.full((self.num_nodes,), -1, dtype=torch.float)

        # --- AQUÍ ES DONDE OCURRE LA MAGIA ---
        if ordering_strategy == 'random':
            self.node_order = torch.randperm(self.num_nodes)
        
        elif ordering_strategy == 'degree_high_low':
            # gt(0) crea una matriz booleana de adyacencia, sum(dim=1) calcula el grado
            degrees = self.weights_matrix.gt(0).sum(dim=1)
            self.node_order = torch.argsort(degrees, descending=True)

        elif ordering_strategy == 'degree_low_high':
            degrees = self.weights_matrix.gt(0).sum(dim=1)
            self.node_order = torch.argsort(degrees, descending=False)
            
        elif ordering_strategy == 'weighted_degree_high_low':
            # Simplemente sumamos los pesos de las aristas para cada nodo
            weighted_degrees = self.weights_matrix.sum(dim=1)
            self.node_order = torch.argsort(weighted_degrees, descending=True)

        else:
            self.node_order = torch.randperm(self.num_nodes)

        self.static_degrees = self.weights_matrix.gt(0).sum(dim=1)
        self.static_weighted_degrees = self.weights_matrix.sum(dim=1)

        return self._get_state()

    def _get_state(self):
        """
        Construye la representación del estado actual como objeto Data de PyTorch Geometric.

        Returns:
            Data: Objeto Data con x (node features), edge_index, y edge_attr.
        """
        # Componente Dinámico: Matriz de características de nodos X_t
        # 
        # El nuevo vector de características para cada nodo tendrá 8 dimensiones:
        # 0: Indicador: No Asignado
        # 1: Indicador: En Partición 0
        # 2: Indicador: En Partición 1
        # 3: Indicador: Es el nodo a decidir ahora
        # 4: Estática: Grado del nodo
        # 5: Estática: Suma de pesos de aristas incidentes
        # 6: Dinámica: Suma de pesos a nodos en Partición 0
        # 7: Dinámica: Suma de pesos a nodos en Partición 1

        num_features = 8  # Número de características por nodo

        node_features = torch.zeros(self.num_nodes, num_features)

        if not self.is_done():
            node_to_decide_idx = self.node_order[self.current_step].item()
        else:
            node_to_decide_idx = -1

        assigned_to_0_mask = (self.current_partition == 0)
        assigned_to_1_mask = (self.current_partition == 1)


        for i in range(self.num_nodes):
            # Característica 0-2: Estado de asignación (one-hot)
            assignment = self.current_partition[i]
            if assignment == -1:
                node_features[i, 0] = 1
            elif assignment == 0:
                node_features[i, 1] = 1
            else: # == 1
                node_features[i, 2] = 1
                
            # Característica 3: Indicador del vértice a decidir
            if i == node_to_decide_idx:
                node_features[i, 3] = 1

            # Característica 4-5: Propiedades estáticas (pre-calculadas)
            node_features[i, 4] = self.static_degrees[i]
            node_features[i, 5] = self.static_weighted_degrees[i]

            # Característica 6-7: Conectividad con particiones
            # Suma de pesos de las aristas desde el nodo `i` a nodos en la partición 0
            node_features[i, 6] = self.weights_matrix[i, assigned_to_0_mask].sum()
            # Suma de pesos de las aristas desde el nodo `i` a nodos en la partición 1
            node_features[i, 7] = self.weights_matrix[i, assigned_to_1_mask].sum()
 
        return Data(
            x=node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr
        )

    def step(self, action: int):
        """
        Avanza un paso en el entorno aplicando una acción.

        Args:
            action (int): La acción a tomar (0 para partición 0, 1 para partición 1).

        Returns:
            tuple: (next_state, reward, done) donde next_state es un objeto Data.
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