# drl_environment.py

import torch
import numpy as np
import random
# Asumimos que las clases de MAXCUT.py están disponibles
from MAXCUT import MaxCutInstance, MaxCutState
from drl_features import state_to_graph

class MaxCutDRLEnv:
    """
    Un entorno para el problema de Max-Cut con un enfoque constructivo,
    diseñado para ser compatible con algoritmos de RL.
    """
    def __init__(self, instance: MaxCutInstance):
        self.instance = instance
        self.num_nodes = instance.num_nodes
        self.state = None
        self.node_order = None
        self.current_step = 0

    def reset(self, order_strategy='random'):
        """
        Reinicia el entorno a un estado inicial. 
        
        Args:
            order_strategy (str): La estrategia para ordenar los nodos.
                'random': Orden aleatorio en cada reseteo (comportamiento anterior). 
                'degree_desc': Ordenado por grado de nodo, de mayor a menor.
                'sequential': Orden secuencial por índice (0, 1, 2, ...).
        
        Returns:
            tuple: La representación inicial del grafo (features, edge_index, edge_weight). 
        """
        self.state = MaxCutState(self.instance)
        self.node_order = list(range(self.num_nodes))

        # --- NUEVA LÓGICA DE ORDENAMIENTO ---
        if order_strategy == 'random':
            random.shuffle(self.node_order)
        elif order_strategy == 'degree_desc':
            # Calcular grados de los nodos (número de aristas no nulas)
            degrees = np.sum(self.instance.weights_matrix != 0, axis=1)
            # Ordenar la lista de nodos basada en los grados, de forma descendente
            self.node_order = sorted(self.node_order, key=lambda n: degrees[n], reverse=True)
        elif order_strategy == 'sequential':
            # No hacer nada, ya está en orden secuencial por defecto
            pass
        else:
            raise ValueError(f"Estrategia de ordenamiento desconocida: {order_strategy}")

        self.current_step = 0
        
        # El estado inicial tiene el primer nodo de la secuencia listo para ser asignado
        return state_to_graph(self.state)

    def step(self, action: int):
        """
        Ejecuta un paso en el entorno. 
        
        Args:
            action (int): La partición a la que asignar el nodo actual (0 o 1). 

        Returns:
            tuple: Una tupla conteniendo:
                - next_graph_repr (tuple): Representación del siguiente estado. 
                - reward (float): Recompensa obtenida en este paso. 
                - done (bool): True si el episodio ha terminado. 
        """
        # 1. Obtener el nodo a asignar en este paso
        node_to_assign = self.node_order[self.current_step]
        
        # 2. Calcular la recompensa INMEDIATA (antes de actualizar el estado)
        # La recompensa es la suma de los pesos de las aristas recién cortadas. 
        reward = 0.0
        # Iteramos sobre los nodos YA ASIGNADOS
        for i in range(self.num_nodes):
            if self.state.partition[i] != -1:
                # Si el vecino 'i' está en la partición opuesta a la acción que vamos a tomar
                if self.state.partition[i] != action:
                    reward += self.instance.get_edge_weight(node_to_assign, i)
        
        # 3. Actualizar el estado con la acción
        self.state.partition[node_to_assign] = action
        if self.state.unassigned:
            self.state.unassigned.remove(node_to_assign)

        # 4. Preparar el siguiente estado
        self.current_step += 1
        done = self.current_step == self.num_nodes
        if done:
             self.state.is_complete = True
        
        self.state.cut_value = self.state.update_cut_value()
        next_graph_repr = state_to_graph(self.state)
        
        return next_graph_repr, reward, done
        
    def get_current_node_to_assign(self):
        """Devuelve el índice del nodo que se debe asignar en el paso actual."""
        if self.current_step < self.num_nodes:
            return self.node_order[self.current_step]
        return None