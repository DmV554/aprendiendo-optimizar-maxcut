# drl_agent.py

import torch
# Asumimos que las siguientes clases y funciones están disponibles
from drl_model import ActorCriticGAT,ActorCriticGAT_V2,ActorCriticGAT_V3
from drl_features import state_to_graph
from MAXCUT import MaxCutState, MaxCutEnvironment

class DRLAgent:
    def __init__(self, model_path, device='cpu'):
        """
        Un agente que utiliza un modelo DRL entrenado para tomar decisiones constructivas.

        Args:
            model_path (str): Ruta al archivo .pth del modelo entrenado.
            device (str): Dispositivo en el que correr el modelo ('cpu' o 'cuda').
        """
        self.device = torch.device(device)
        self.model = ActorCriticGAT_V2().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Poner el modelo en modo de evaluación
        
        self.node_order = None
        self.current_step = 0

    def _initialize_episode(self, state: MaxCutState):
        """Prepara el agente para un nuevo episodio/instancia."""
        self.node_order = list(range(state.inst_info.num_nodes))
        # Para evaluación, usamos un orden fijo y determinista.
        # random.shuffle(self.node_order) # Descomentar para orden aleatorio
        self.current_step = 0
        
    def reset(self):
        """
        Reinicia el estado interno del agente para resolver una nueva instancia.
        """
        self.node_order = None
        self.current_step = 0

    def action_policy(self, state: MaxCutState, env: MaxCutEnvironment):
        """
        Define la política de acción del agente DRL para evaluación.
        Elige la acción de forma determinista (greedy) según la política aprendida.

        Args:
            state (MaxCutState): El estado actual.
            env (MaxCutEnvironment): El entorno (no se usa directamente, pero es parte de la interfaz).

        Returns:
            tuple or None: La acción constructiva a tomar, o None si la solución está completa.
        """
        if state.is_complete:
            return None
        
        # Si es el primer paso de una instancia, inicializar el orden de nodos
        if self.current_step == 0 or self.node_order is None:
             self._initialize_episode(state)

        # 1. Obtener el estado actual en formato de grafo
        graph_repr = state_to_graph(state)
        x, edge_index, edge_weight = graph_repr
        x, edge_index, edge_weight = x.to(self.device), edge_index.to(self.device), edge_weight.to(self.device)

        # 2. Obtener el nodo a decidir
        node_to_assign = self.node_order[self.current_step]
        
        # 3. Obtener logits del modelo
        with torch.no_grad():
            logits, _ = self.model(x, edge_index, edge_weight, node_to_assign)
        
        # 4. Seleccionar la mejor acción de forma determinista (greedy)
        best_action = torch.argmax(logits).item()
        
        # 5. Incrementar el contador de pasos para el siguiente llamado
        self.current_step += 1
        
        # 6. Devolver la acción en el formato esperado por SingleAgentSolver
        return ("constructive", node_to_assign, best_action)