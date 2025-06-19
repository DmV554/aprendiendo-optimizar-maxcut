import numpy as np
import random
from copy import deepcopy


class MaxCutInstance:
    def __init__(self, weights_matrix):
        """
        Inicializa una instancia del problema Max-Cut.

        Parameters:
        weights_matrix (np.ndarray): Matriz de adyacencia con pesos de las aristas.
                                    weights_matrix[i][j] es el peso de la arista entre i y j.
        """
        self.weights_matrix = weights_matrix
        self.num_nodes = len(weights_matrix)

    def get_edge_weight(self, i, j):
        """Retorna el peso de la arista entre los nodos i y j."""
        return self.weights_matrix[i][j]
        
    def evaluate_partition(self, partition):
        """
        Calcula el valor del corte para una partición dada.
        
        Parameters:
        partition (list): Lista de 0s y 1s que representa la partición.
        
        Returns:
        float: Valor del corte.
        """
        cut_value = 0
        
        # Solo consideramos nodos que ya han sido asignados (no -1)
        assigned = [i for i in range(self.num_nodes) if partition[i] != -1]
        
        for i in assigned:
            for j in assigned:
                if i < j and partition[i] != partition[j]:
                    cut_value += self.get_edge_weight(i, j)
                    
        return cut_value


class MaxCutState:
    def __init__(self, inst_info, partition=None):
        """
        Inicializa un estado del problema Max-Cut.

        Parameters:
        inst_info (MaxCutInstance): Instancia del problema Max-Cut.
        partition (list, optional): Lista de 0s y 1s que representa la partición actual.
                                    0 significa que el nodo está en el conjunto A,
                                    1 significa que el nodo está en el conjunto B.
        """
        self.inst_info = inst_info

        if partition is None:
            # Inicialmente, no hay nodos asignados (-1)
            self.partition = [-1] * inst_info.num_nodes
            self.unassigned = set(range(inst_info.num_nodes))
        else:
            self.partition = partition
            self.unassigned = set(i for i in range(inst_info.num_nodes) if partition[i] == -1)

        self.is_complete = len(self.unassigned) == 0
        self.cut_value = self.update_cut_value()

    def update_cut_value(self):
        """
        Calcula el valor del corte actual, que es la suma de los pesos de las aristas
        que conectan nodos en diferentes conjuntos.
        """
        cut_value = 0

        # Solo consideramos nodos que ya han sido asignados
        assigned = [i for i in range(self.inst_info.num_nodes) if self.partition[i] != -1]

        for i in assigned:
            for j in assigned:
                if i < j and self.partition[i] != self.partition[j]:
                    cut_value += self.inst_info.get_edge_weight(i, j)

        return cut_value

    def __deepcopy__(self, memo):
        """
        Método para realizar una copia profunda del estado,
        manteniendo la referencia a inst_info.
        """
        return MaxCutState(
            inst_info=self.inst_info,  # Por referencia
            partition=deepcopy(self.partition)
        )


class MaxCutEnvironment:
    @staticmethod
    def gen_actions(state, type, shuffle=False):
        """
        Genera las acciones posibles a partir de un estado dado.

        Parameters:
        state (MaxCutState): Estado actual.
        type (str): Tipo de acción ("constructive", "flip", "pair_swap", "pair_flip").
        shuffle (bool): Indica si se deben mezclar aleatoriamente las acciones.

        Returns:
        generator: Generador de acciones posibles.
        """
        actions = []  # Inicializamos fuera de los ifs

        if type == "constructive":
            # Para acciones constructivas, asignamos un nodo no asignado a A (0) o B (1)
            if state.is_complete:
                # No hay acciones constructivas si la partición está completa
                pass
            else:
                for node in state.unassigned:
                    actions.append(("constructive", node, 0))  # Asignar a conjunto A
                    actions.append(("constructive", node, 1))  # Asignar a conjunto B

        elif type == "flip":
            # Para acciones de búsqueda local, cambiamos un nodo de conjunto
            if not state.is_complete:
                # No podemos hacer flip en un estado incompleto
                pass
            else:
                for node in range(state.inst_info.num_nodes):
                    # Cambiamos el nodo de conjunto (de 0 a 1 o de 1 a 0)
                    actions.append(("flip", node))

        # --- NUEVO: Movimiento Pair Swap ---
        elif type == "pair_swap":
            # Para acciones de intercambio de pares entre conjuntos diferentes
            if not state.is_complete:
                # No podemos hacer swap en un estado incompleto
                pass
            else:
                nodes_in_0 = [i for i in range(state.inst_info.num_nodes) if state.partition[i] == 0]
                nodes_in_1 = [i for i in range(state.inst_info.num_nodes) if state.partition[i] == 1]

                # Generar todas las combinaciones posibles de nodos entre conjuntos diferentes
                for u in nodes_in_0:
                    for v in nodes_in_1:
                        actions.append(("pair_swap", u, v))
        # --- FIN NUEVO ---

        # --- NUEVO: Movimiento Pair Flip ---
        elif type == "pair_flip":
            # Para acciones de cambio de par de vértices del mismo conjunto
            if not state.is_complete:
                # No podemos hacer flip en un estado incompleto
                pass
            else:
                nodes_in_0 = [i for i in range(state.inst_info.num_nodes) if state.partition[i] == 0]
                nodes_in_1 = [i for i in range(state.inst_info.num_nodes) if state.partition[i] == 1]

                # Generar todas las combinaciones posibles dentro del mismo conjunto
                # Pares del conjunto 0
                for i in range(len(nodes_in_0)):
                    for j in range(i + 1, len(nodes_in_0)):
                        actions.append(("pair_flip", nodes_in_0[i], nodes_in_0[j]))

                # Pares del conjunto 1
                for i in range(len(nodes_in_1)):
                    for j in range(i + 1, len(nodes_in_1)):
                        actions.append(("pair_flip", nodes_in_1[i], nodes_in_1[j]))
        # --- FIN NUEVO ---

        else:
            raise NotImplementedError(f"Tipo de acción '{type}' no implementado")

        if shuffle:
            random.shuffle(actions)

        # Convertimos a generador al final
        for action in actions:
            yield action

    @staticmethod
    def state_transition(state, action):
        """
        Aplica una acción a un estado y retorna el nuevo estado modificado.

        Parameters:
        state (MaxCutState): Estado actual (será modificado).
        action (tuple): Acción a aplicar.

        Returns:
        MaxCutState: El mismo estado modificado después de aplicar la acción.
        """
        action_type = action[0]

        # Para acciones constructivas
        if action_type == "constructive" and not state.is_complete:
            node, assignment = action[1], action[2]
            if node in state.unassigned:
                state.partition[node] = assignment
                state.unassigned.remove(node)
                if len(state.unassigned) == 0:
                    state.is_complete = True
                state.cut_value = state.update_cut_value()  # Recalculo completo necesario aquí

        # Para acciones de búsqueda local (flip)
        elif action_type == "flip" and state.is_complete:
            node = action[1]
            # Calculamos el delta ANTES de cambiar el estado
            delta = MaxCutEnvironment._calculate_delta_flip(state, node)
            # Invertimos la asignación del nodo (de 0 a 1 o de 1 a 0)
            state.partition[node] = 1 - state.partition[node]
            # Actualizamos el valor del corte eficientemente
            state.cut_value += delta

        # --- NUEVO: Transición para Pair Swap ---
        elif action_type == "pair_swap" and state.is_complete:
            u, v = action[1], action[2]
            # Verificamos que efectivamente estén en diferentes conjuntos
            if state.partition[u] != state.partition[v]:
                # Calculamos el delta ANTES de cambiar el estado
                delta = MaxCutEnvironment._calculate_delta_pair_swap(state, u, v)
                # Intercambiamos las asignaciones
                state.partition[u] = 1 - state.partition[u]
                state.partition[v] = 1 - state.partition[v]
                # Actualizamos el valor del corte eficientemente
                state.cut_value += delta
        # --- FIN NUEVO ---

        # --- NUEVO: Transición para Pair Flip ---
        elif action_type == "pair_flip" and state.is_complete:
            u, v = action[1], action[2]
            # Verificamos que efectivamente estén en el mismo conjunto
            if state.partition[u] == state.partition[v]:
                # Calculamos el delta ANTES de cambiar el estado
                delta = MaxCutEnvironment._calculate_delta_pair_flip(state, u, v)
                # Movemos ambos nodos al otro conjunto
                state.partition[u] = 1 - state.partition[u]
                state.partition[v] = 1 - state.partition[v]
                # Actualizamos el valor del corte eficientemente
                state.cut_value += delta
        # --- FIN NUEVO ---

        elif (action_type in ["flip", "pair_swap", "pair_flip"] and not state.is_complete) or \
                (action_type == "constructive" and state.is_complete):
            # Ignorar acciones inválidas para el estado actual (e.g., flip en incompleto)
            pass
        else:
            raise NotImplementedError(
                f"Acción '{action}' ('{action_type}') no válida o no implementada para este estado")

        return state

    @staticmethod
    def _calculate_delta_flip(state, node):
        """Calcula el cambio (delta) en el cut value si se hace flip al nodo."""
        if state.partition[node] == -1: return 0  # No se puede hacer flip a un nodo no asignado

        delta = 0
        current_assignment = state.partition[node]
        for i in range(state.inst_info.num_nodes):
            # Solo considerar vecinos asignados y distintos al nodo
            if i != node and state.partition[i] != -1:
                weight = state.inst_info.get_edge_weight(i, node)
                if state.partition[i] == current_assignment:
                    # Antes estaban en el mismo conjunto (no cortaba), ahora estarán en diferentes (cortará)
                    delta += weight
                else:
                    # Antes estaban en diferentes conjuntos (cortaba), ahora estarán en el mismo (no cortará)
                    delta -= weight
        return delta

    @staticmethod
    def _calculate_delta_pair_swap(state, u, v):
        """Calcula el delta en el cut value si se intercambian u y v (de conjuntos distintos)."""
        if state.partition[u] == state.partition[v] or state.partition[u] == -1 or state.partition[v] == -1:
            # No tiene sentido si están en el mismo conjunto o no asignados
            return 0

        # El cambio total es la suma de los cambios individuales si cada uno hiciera flip,
        # más un ajuste por la arista (u,v) si existe.
        # Si u hace flip, el estado de (u,v) pasa de cortar a no cortar (delta -= Wuv).
        # Si v hace flip, el estado de (u,v) pasa de cortar a no cortar (delta -= Wuv).
        # Como ambos se mueven, el estado final de (u,v) es cortado (igual que al inicio).
        # El cambio neto debido a la arista (u,v) es 0. Sin embargo, en los deltas individuales
        # se restó Wuv dos veces. Debemos sumar 2 * Wuv para compensar.
        delta_u = MaxCutEnvironment._calculate_delta_flip(state, u)
        delta_v = MaxCutEnvironment._calculate_delta_flip(state, v)
        weight_uv = state.inst_info.get_edge_weight(u, v)

        return delta_u + delta_v + 2 * weight_uv

    @staticmethod
    def _calculate_delta_pair_flip(state, u, v):
        """Calcula el delta en el cut value si u y v cambian al otro conjunto (del mismo conjunto)."""
        if state.partition[u] != state.partition[v] or state.partition[u] == -1:
            # No tiene sentido si están en conjuntos distintos o no asignados
            return 0

        # El cambio total es la suma de los cambios individuales si cada uno hiciera flip,
        # menos un ajuste por la arista (u,v) si existe.
        # Si u hace flip, el estado de (u,v) pasa de no cortar a cortar (delta += Wuv).
        # Si v hace flip, el estado de (u,v) pasa de no cortar a cortar (delta += Wuv).
        # Como ambos se mueven, el estado final de (u,v) es no cortado (igual que al inicio).
        # El cambio neto debido a la arista (u,v) es 0. Sin embargo, en los deltas individuales
        # se sumó Wuv dos veces. Debemos restar 2 * Wuv para compensar.
        delta_u = MaxCutEnvironment._calculate_delta_flip(state, u)
        delta_v = MaxCutEnvironment._calculate_delta_flip(state, v)
        weight_uv = state.inst_info.get_edge_weight(u, v)

        return delta_u + delta_v - 2 * weight_uv

    @staticmethod
    def calculate_cut_value_after_action(state, action):
        """
        Calcula el valor del corte después de aplicar una acción sin modificar el estado.
        Utiliza cálculos de delta eficientes para 'flip', 'pair_swap' y 'pair_flip'.

        Parameters:
        state (MaxCutState): Estado actual.
        action (tuple): Acción a aplicar.

        Returns:
        float: Valor del corte después de aplicar la acción.
        """
        action_type = action[0]

        if action_type == "flip":
            if not state.is_complete: return state.cut_value  # No cambia si no es completo
            node = action[1]
            delta = MaxCutEnvironment._calculate_delta_flip(state, node)
            return state.cut_value + delta

        # --- NUEVO: Cálculo eficiente para Pair Swap ---
        elif action_type == "pair_swap":
            if not state.is_complete: return state.cut_value
            u, v = action[1], action[2]
            if state.partition[u] == state.partition[v]: return state.cut_value  # Acción inválida
            delta = MaxCutEnvironment._calculate_delta_pair_swap(state, u, v)
            return state.cut_value + delta
        # --- FIN NUEVO ---

        # --- NUEVO: Cálculo eficiente para Pair Flip ---
        elif action_type == "pair_flip":
            if not state.is_complete: return state.cut_value
            u, v = action[1], action[2]
            if state.partition[u] != state.partition[v]: return state.cut_value  # Acción inválida
            delta = MaxCutEnvironment._calculate_delta_pair_flip(state, u, v)
            return state.cut_value + delta
        # --- FIN NUEVO ---

        # Para acciones constructivas o no implementadas de forma eficiente:
        # (Mantenemos el deepcopy como fallback, aunque para constructivas
        # podría optimizarse también si fuera necesario)
        else:
            # Creamos una copia del estado, aplicamos la acción y calculamos el valor del corte
            try:
                state_copy = deepcopy(state)
                # Usamos state_transition que AHORA usa deltas internamente para movimientos
                # de búsqueda local, pero recalcula completo para constructivas.
                state_copy = MaxCutEnvironment.state_transition(state_copy, action)
                return state_copy.cut_value
            except NotImplementedError:
                # Si la acción no es válida para el estado copia, retorna el valor actual
                return state.cut_value
            except ValueError:  # Podría ocurrir si la copia falla o algo inesperado
                return state.cut_value


def evalConstructiveActions(state, env):
    """
    Evalúa las acciones constructivas para el estado actual.

    Parameters:
    state (MaxCutState): Estado actual.
    env (MaxCutEnvironment): Ambiente del problema.

    Returns:
    list: Lista de tuplas (acción, evaluación).
    """
    evals = []
    for action in env.gen_actions(state, "constructive"):
        # Calculamos el valor del corte después de aplicar la acción
        eval_value = env.calculate_cut_value_after_action(state, action)
        evals.append((action, eval_value))
    return evals


def generate_random_instance(n_nodes, min_weight=1, max_weight=100, density=0.8):
    """
    Genera una instancia aleatoria del problema Max-Cut de forma eficiente usando NumPy.

    Parameters:
    n_nodes (int): Número de nodos.
    min_weight (int): Peso mínimo de las aristas.
    max_weight (int): Peso máximo de las aristas.
    density (float): Densidad del grafo (probabilidad de que exista una arista).

    Returns:
    np.ndarray: Matriz de pesos (adyacencia ponderada) de la instancia.
    """
    # 1. Crear máscara de densidad para la triangular superior
    # Genera números aleatorios [0, 1) para cada posible arista en la triangular superior
    upper_tri_mask = np.random.rand(n_nodes, n_nodes) < density
    # Asegura que sea estrictamente triangular superior (excluye diagonal)
    upper_tri_mask = np.triu(upper_tri_mask, k=1)

    # 2. Generar pesos aleatorios para toda la matriz
    # Es más simple generar para toda la matriz y luego aplicar la máscara
    weights = np.random.uniform(min_weight, max_weight, size=(n_nodes, n_nodes))

    # 3. Aplicar la máscara a los pesos (solo en la triangular superior)
    weights_matrix = np.zeros((n_nodes, n_nodes))
    weights_matrix[upper_tri_mask] = weights[upper_tri_mask]

    # 4. Hacer la matriz simétrica copiando la triangular superior a la inferior
    weights_matrix = weights_matrix + weights_matrix.T

    # 5. Asegurar que la diagonal sea cero (opcional pero común en MaxCut)
    # np.fill_diagonal(weights_matrix, 0) # Ya debería ser cero por cómo se construyó

    return weights_matrix