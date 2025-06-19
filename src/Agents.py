from collections import deque
from copy import deepcopy
import math
import random

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

class GreedyAgent:
    def __init__(self, eval_actions=None):
        """
        Inicializa un agente greedy.

        Parameters:
        eval_actions (function): Función para evaluar las acciones (opcional).
        """
        self.eval_actions = eval_actions

    def select_action(self, *args):
        """
        Selecciona la acción con la mejor evaluación.
        Puede ser llamado de dos formas:
        1. select_action(evals) - donde evals es una lista de tuplas (acción, evaluación)
        2. select_action(env, state) - donde env es el entorno y state es el estado

        Returns:
        tuple: Acción seleccionada.
        """
        # Detectar el tipo de llamada
        if len(args) == 1 and isinstance(args[0], list):
            # Caso 1: se pasó directamente la lista de evaluaciones
            evals = args[0]
        elif len(args) == 2:
            # Caso 2: se pasó el entorno y el estado
            env, state = args
            # Usar evalConstructiveActions si no se especificó una función de evaluación
            if self.eval_actions is None:
                evals = evalConstructiveActions(state, env)
            else:
                evals = self.eval_actions(state, env)
        else:
            raise ValueError("Número incorrecto de argumentos para select_action")
            
        if not evals:
            return None
            
        return max(evals, key=lambda x: x[1])[0]

    def action_policy(self, state, env):
        """
        Define la política de acción del agente.

        Parameters:
        state (MaxCutState): Estado actual.
        env (MaxCutEnvironment): Ambiente del problema.

        Returns:
        tuple: Acción seleccionada.
        """
        evals = self.eval_actions(state, env)
        if len(evals) == 0:
            return None
        return self.select_action(evals)


class FirstImprovementAgent:
    # MODIFICADO: Añadido __init__ y neighborhood_type
    def __init__(self, neighborhood_type="flip"):
        """
        Inicializa un agente de primera mejora.

        Parameters:
        neighborhood_type (str): Tipo de vecindario a explorar ("flip", "pair_swap", etc.)
        """
        self.neighborhood_type = neighborhood_type

    def action_policy(self, state, env):
        """
        Define la política de acción del agente. Busca la primera acción
        en el vecindario especificado que mejore la solución actual.

        Parameters:
        state (MaxCutState): Estado actual.
        env (MaxCutEnvironment): Ambiente del problema.

        Returns:
        tuple: Acción seleccionada que mejora, o None si no se encuentra ninguna.
        """
        current_value = state.cut_value

        # MODIFICADO: Usar self.neighborhood_type en lugar de "flip"
        # Iteramos sobre las acciones del tipo especificado, mezcladas aleatoriamente
        for action in env.gen_actions(state, self.neighborhood_type, shuffle=True):
            new_value = env.calculate_cut_value_after_action(state, action)
            if new_value > current_value:  # Buscamos maximizar el corte
                return action # Retorna la primera acción que mejora

        return None  # No se encontró ninguna mejora en este vecindario
        
    def select_action(self, env, state):
        """
        Alias para action_policy para mantener compatibilidad con la interfaz de GreedyAgent.
        
        Parameters:
        env (MaxCutEnvironment): Ambiente del problema.
        state (MaxCutState): Estado actual.
        
        Returns:
        tuple: Acción seleccionada que mejora, o None si no se encuentra ninguna.
        """
        return self.action_policy(state, env)

class SingleAgentSolver:
    def __init__(self, env, agent, max_actions=1000):
        """
        Inicializa un solucionador de un solo agente.

        Parameters:
        env (MaxCutEnvironment): Ambiente del problema.
        agent (object): Agente que resolverá el problema.
        max_actions (int): Número máximo de acciones.
        """
        self.env = env
        self.agent = agent
        self.max_actions = max_actions

    def solve(self, state):
        """
        Resuelve el problema a partir de un estado inicial.

        Parameters:
        state (MaxCutState): Estado inicial.

        Returns:
        MaxCutState: Estado final.
        """
        n_actions = 0
        while n_actions < self.max_actions:
            action = self.agent.action_policy(state, self.env)
            if action is None:
                break
            state = self.env.state_transition(state, action)
            n_actions += 1
        return state

class Perturbation:
    def __init__(self, env, type, pert_size=3):
        """
        Inicializa un perturbador para el algoritmo ILS.

        Parameters:
        env (MaxCutEnvironment): Ambiente del problema.
        type (str): Tipo de acción para perturbar.
        pert_size (int): Tamaño de la perturbación.
        """
        self.env = env
        self.type = type
        self.pert_size = pert_size

    def __call__(self, state):
        """
        Aplica la perturbación a un estado.

        Parameters:
        state (MaxCutState): Estado a perturbar.

        Returns:
        MaxCutState: Estado perturbado.
        """
        state_copy = deepcopy(state)
        for _ in range(self.pert_size):
            actions = list(self.env.gen_actions(state_copy, self.type, shuffle=True))
            if actions:
                action = actions[0]  # Tomamos la primera acción
                state_copy = self.env.state_transition(state_copy, action)
        return state_copy

class DefaultAcceptanceCriterion:
    def __call__(self, best_value, new_value):
        """
        Evalúa si se debe aceptar una nueva solución.

        Parameters:
        best_value (float): Valor de la mejor solución.
        new_value (float): Valor de la nueva solución.

        Returns:
        bool: True si se debe aceptar la nueva solución, False en caso contrario.
        """
        return new_value >= best_value  # Para maximización

class ILS:
    def __init__(self, local_search, perturbation, acceptance_criterion=DefaultAcceptanceCriterion(), max_iterations=50):
        """
        Inicializa un algoritmo ILS.

        Parameters:
        local_search (SingleAgentSolver): Solucionador de búsqueda local.
        perturbation (Perturbation): Perturbador.
        acceptance_criterion (function): Criterio de aceptación.
        max_iterations (int): Número máximo de iteraciones.
        """
        self.local_search = local_search
        self.perturbation = perturbation
        self.acceptance_criterion = acceptance_criterion
        self.max_iterations = max_iterations

    def solve(self, initial_solution):
        """
        Resuelve el problema a partir de una solución inicial.

        Parameters:
        initial_solution (MaxCutState): Solución inicial.

        Returns:
        MaxCutState: Mejor solución encontrada.
        """
        current_solution = initial_solution
        current_solution = self.local_search.solve(current_solution)
        best_solution = deepcopy(current_solution)
        best_solution_value = best_solution.cut_value

        for _ in range(self.max_iterations):
            # Perturbar la solución actual para escapar de óptimos locales
            perturbed_solution = self.perturbation(current_solution)

            # Aplicar búsqueda local en la solución perturbada
            local_optimum = self.local_search.solve(perturbed_solution)
            value = local_optimum.cut_value

            # Decidir si se acepta la nueva solución
            if self.acceptance_criterion(best_solution_value, value):
                current_solution = local_optimum
                if value > best_solution_value:  # Para maximización
                    best_solution = deepcopy(current_solution)
                    best_solution_value = value

        return best_solution


class SimulatedAnnealingAgent:
    """
    Agente que implementa el algoritmo de Simulated Annealing para el problema Max-Cut.
    Selecciona una acción aleatoria del vecindario especificado en cada paso.
    """
    # MODIFICADO: Añadido neighborhood_type al __init__
    def __init__(self, initial_temperature=100.0, cooling_rate=0.95, iterations_per_temp=100, min_temperature=0.1, neighborhood_type="flip"):
        """
        Inicializa el agente de Simulated Annealing.

        Parameters:
        initial_temperature (float): Temperatura inicial.
        cooling_rate (float): Tasa de enfriamiento (< 1).
        iterations_per_temp (int): Iteraciones por nivel de temperatura.
        min_temperature (float): Temperatura mínima de parada.
        neighborhood_type (str): Tipo de vecindario del cual seleccionar acciones
                                 aleatoriamente ("flip", "pair_swap", etc.).
        """
        # ... (asignaciones de otros parámetros como antes) ...
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.min_temperature = min_temperature
        self.neighborhood_type = neighborhood_type # Guardar el tipo de vecindario

        self.temperature = initial_temperature
        self.iteration_count = 0

    def metropolis_criterion(self, delta, temperature):
        # ... (sin cambios en este método) ...
        if delta > 0:
            return True
        if temperature <= 0: # Evitar división por cero o exp(inf)
             return False
        try:
            acceptance_probability = math.exp(delta / temperature)
        except OverflowError:
            acceptance_probability = 0 # Si delta es muy negativo, la prob es esencialmente 0
        return random.random() < acceptance_probability


    def action_policy(self, state, env):
        """
        Determina la siguiente acción a realizar según la política de Simulated Annealing,
        seleccionando aleatoriamente del vecindario especificado.

        Parameters:
        state (MaxCutState): Estado actual.
        env (MaxCutEnvironment): Ambiente del problema.

        Returns:
        tuple or None: Acción a realizar, o None si no hay acción disponible o la temp es muy baja.
        """
        if not state.is_complete:
            raise ValueError("El estado para Simulated Annealing debe estar completo")

        # Condición de parada por temperatura mínima
        if self.temperature < self.min_temperature:
             # print("SA: Temperatura mínima alcanzada.") # Opcional: Log
             return None

        # Reiniciar si es necesario (por ejemplo, si se llama repetidamente fuera de un solver)
        # Esto podría necesitar ajuste dependiendo de cómo se use el agente.
        # Si se usa con SingleAgentSolver, este reinicio podría no ser ideal.
        # Considera manejar el estado de la temperatura externamente si es necesario.
        # Por ahora, mantenemos el reinicio simple basado en iteraciones.
        # if self.iteration_count == 0:
        #    self.temperature = self.initial_temperature

        # MODIFICADO: Usar self.neighborhood_type
        # Generar posibles acciones del tipo especificado
        # Usamos list() para poder usar random.choice
        actions = list(env.gen_actions(state, self.neighborhood_type, shuffle=False)) # No necesitamos shuffle aquí si escogemos al azar

        if not actions:
            # print(f"SA: No hay acciones en el vecindario {self.neighborhood_type}.") # Opcional: Log
            return None # No hay acciones posibles

        # Seleccionar una acción aleatoria del vecindario especificado
        action = random.choice(actions)

        # Calcular el cambio en el valor del corte
        new_cut_value = env.calculate_cut_value_after_action(state, action)
        delta = new_cut_value - state.cut_value

        # Decidir si aceptar la acción usando el criterio de Metropolis
        accepted = self.metropolis_criterion(delta, self.temperature)

        # Actualizar contador y temperatura SIEMPRE que se evalúa una acción,
        # no solo si se acepta (esto es común en SA).
        self.iteration_count += 1
        if self.iteration_count >= self.iterations_per_temp:
            self.temperature *= self.cooling_rate
            self.iteration_count = 0 # Reiniciar contador para el siguiente nivel de temp.
            # print(f"SA: Nueva temperatura: {self.temperature:.2f}") # Opcional: Log

        if accepted:
            return action # Devolvemos la acción aceptada
        else:
            # Si no aceptamos la acción, en el contexto de SingleAgentSolver,
            # devolver None significa que el estado no cambia en esta iteración,
            # pero la temperatura sí se actualizó. Esto está bien para SA.
            return None


class TabuSearchSolver:
    """
    Implementa el algoritmo de Búsqueda Tabú para el problema Max-Cut.
    Utiliza una lista tabú basada en los nodos modificados recientemente (vecindario 'flip').
    """
    def __init__(self, env, initial_solution, max_iterations=1000, tabu_tenure=10, neighborhood_type="flip"):
        """
        Inicializa el solucionador de Búsqueda Tabú.

        Parameters:
        env (MaxCutEnvironment): Ambiente del problema.
        initial_solution (MaxCutState): Estado inicial desde donde comenzar la búsqueda.
                                        Debe ser una solución completa (is_complete=True).
        max_iterations (int): Número máximo de iteraciones que realizará el algoritmo.
        tabu_tenure (int): Duración (en iteraciones) que un atributo (nodo) permanece tabú
                           después de ser modificado.
        neighborhood_type (str): Tipo de vecindario a explorar (por ahora, principalmente "flip").
                                  Se pueden añadir otros como "pair_swap", "pair_flip".
        """
        if not initial_solution.is_complete:
            raise ValueError("La solución inicial para Tabu Search debe estar completa.")

        self.env = env
        self.initial_solution = initial_solution
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.neighborhood_type = neighborhood_type

        # La lista tabú almacenará tuplas (atributo_tabu, iteracion_expira)
        # Para "flip", el atributo es el índice del nodo.
        # Usamos deque por eficiencia al añadir/quitar por ambos extremos si fuera necesario,
        # aunque aquí la usaremos más como una lista con chequeo.
        self.tabu_list = deque()
        # Podríamos usar un diccionario: self.tabu_status = {node_index: expires_at_iteration}

    def _get_attribute_from_action(self, action):
        """ Extrae el atributo relevante de una acción para hacerlo tabú. """
        action_type = action[0]
        if action_type == "flip":
            return action[1] # El atributo tabú es el nodo que se voltea
        elif action_type == "pair_swap" or action_type == "pair_flip":
            # Para estos movimientos, podríamos hacer tabú ambos nodos,
            # o el par (u, v), o definirlo de otra forma.
            # Por simplicidad, hagamos tabú ambos nodos individualmente.
            return (action[1], action[2])
        # Añadir otros tipos de acción si es necesario
        return None

    def _is_tabu(self, attribute, current_iteration):
        """ Verifica si un atributo está actualmente tabú. """
        # Implementación con deque: buscar si el atributo está en la lista
        # y si su tiempo de expiración aún no ha pasado.
        # (Una implementación con diccionario sería más directa:
        # return attribute in self.tabu_status and current_iteration < self.tabu_status[attribute])

        nodes_to_check = []
        if isinstance(attribute, tuple): # Para pair_swap/pair_flip
            nodes_to_check.extend(attribute)
        elif attribute is not None: # Para flip
             nodes_to_check.append(attribute)
        else:
            return False # Atributo no identificable

        for node in nodes_to_check:
            for tabu_node, expires_at in self.tabu_list:
                 if node == tabu_node and current_iteration < expires_at:
                      return True # El nodo está tabú
        return False # El nodo no está tabú

    def _add_to_tabu(self, attribute, current_iteration):
        """ Añade un atributo a la lista tabú. """
        if attribute is None:
            return

        expires_at = current_iteration + self.tabu_tenure

        nodes_to_add = []
        if isinstance(attribute, tuple): # Para pair_swap/pair_flip
            nodes_to_add.extend(attribute)
        else: # Para flip
             nodes_to_add.append(attribute)

        for node in nodes_to_add:
            # Eliminar entradas antiguas para el mismo nodo si existen
            # (para asegurar que solo la última expiración cuenta)
            self.tabu_list = deque([(n, exp) for n, exp in self.tabu_list if n != node])
            # Añadir la nueva entrada tabú
            self.tabu_list.append((node, expires_at))

        # Opcional: Limitar tamaño máximo de la lista tabú si se usa deque
        # while len(self.tabu_list) > max_tabu_size:
        #    self.tabu_list.popleft() # Elimina el más antiguo

    def _clean_tabu_list(self, current_iteration):
         """ Elimina entradas tabú expiradas (opcional, mejora eficiencia si la lista crece mucho). """
         # No es estrictamente necesario si _is_tabu chequea la iteración de expiración.
         # self.tabu_list = deque([(node, exp) for node, exp in self.tabu_list if current_iteration < exp])
         pass # Por ahora, la comprobación en _is_tabu es suficiente


    def solve(self):
        """
        Ejecuta el algoritmo de Búsqueda Tabú.

        Returns:
        MaxCutState: La mejor solución encontrada durante la búsqueda.
        """
        current_solution = deepcopy(self.initial_solution)
        best_solution = deepcopy(current_solution)
        best_solution_value = best_solution.cut_value

        self.tabu_list.clear() # Asegurarse de que la lista esté vacía al inicio

        for current_iteration in range(1, self.max_iterations + 1):
            self._clean_tabu_list(current_iteration) # Opcional

            best_neighbor_action = None
            best_neighbor_value = -float('inf') # Para maximización
            found_better_than_best = False # Para criterio de aspiración

            # Explorar el vecindario
            # Usamos list() para asegurar que exploramos todos antes de decidir
            neighbors_actions = list(self.env.gen_actions(current_solution, self.neighborhood_type, shuffle=False))

            if not neighbors_actions:
                 print(f"Iteración {current_iteration}: No hay acciones posibles en el vecindario '{self.neighborhood_type}'. Deteniendo.")
                 break # No hay vecinos

            for action in neighbors_actions:
                neighbor_value = self.env.calculate_cut_value_after_action(current_solution, action)
                attribute = self._get_attribute_from_action(action)

                is_move_tabu = self._is_tabu(attribute, current_iteration)

                # Criterio de Aspiración: Si la solución mejora la mejor encontrada hasta ahora,
                # ignoramos el estado tabú.
                aspiration_met = neighbor_value > best_solution_value

                # Determinar si el movimiento es admisible
                is_admissible = (not is_move_tabu) or aspiration_met

                if is_admissible:
                    if neighbor_value > best_neighbor_value:
                        best_neighbor_action = action
                        best_neighbor_value = neighbor_value
                        # Guardamos si este vecino superó al mejor global (para info/log)
                        found_better_than_best = aspiration_met and is_move_tabu
                # Si no es admisible, simplemente lo ignoramos

            # Si no encontramos ningún movimiento admisible (raro, pero posible si todo es tabú y no aspira)
            if best_neighbor_action is None:
                # Podríamos intentar buscar en otro vecindario o simplemente detenernos
                print(f"Iteración {current_iteration}: No se encontró ningún movimiento admisible. Deteniendo.")
                break

            # Realizar el mejor movimiento encontrado
            if found_better_than_best:
                 print(f"Iteración {current_iteration}: Criterio de aspiración cumplido para acción {best_neighbor_action}")

            current_solution = self.env.state_transition(current_solution, best_neighbor_action)
            attribute_to_make_tabu = self._get_attribute_from_action(best_neighbor_action)
            self._add_to_tabu(attribute_to_make_tabu, current_iteration)

            # Actualizar la mejor solución global si es necesario
            if current_solution.cut_value > best_solution_value:
                best_solution = deepcopy(current_solution)
                best_solution_value = current_solution.cut_value
                print(f"Iteración {current_iteration}: Nueva mejor solución encontrada con valor: {best_solution_value}")

            # Opcional: Imprimir progreso cada X iteraciones
            # if current_iteration % 100 == 0:
            #     print(f"Iteración {current_iteration}/{self.max_iterations}, Mejor valor: {best_solution_value}, Valor actual: {current_solution.cut_value}")


        print(f"Búsqueda Tabú finalizada. Mejor valor encontrado: {best_solution_value}")
        return best_solution