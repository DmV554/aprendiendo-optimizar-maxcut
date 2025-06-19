import numpy as np
from MAXCUT import MaxCutInstance, MaxCutState, MaxCutEnvironment, generate_random_instance
from Agents import SingleAgentSolver, GreedyAgent, FirstImprovementAgent, Perturbation, ILS, evalConstructiveActions, SimulatedAnnealingAgent
from copy import deepcopy
import os
import pandas as pd
from FileLoader import load_rud_file_to_instance


def solve_and_compare(instance):
    """
    Resuelve una instancia del problema Max-Cut utilizando diferentes algoritmos.

    Parameters:
    instance (MaxCutInstance): Instancia del problema Max-Cut.

    Returns:
    dict: Resultados de los diferentes algoritmos.
    """
    # Crear ambiente
    env = MaxCutEnvironment

    # Crear estado inicial
    initial_state = MaxCutState(instance)

    # Crear agentes
    greedy_agent = GreedyAgent(evalConstructiveActions)
    local_search_agent = FirstImprovementAgent()
    simulated_annealing_agent = SimulatedAnnealingAgent(
        initial_temperature=100.0,
        cooling_rate=0.95,
        iterations_per_temp=100,
        min_temperature=0.1
    )

    # Crear solvers
    greedy_solver = SingleAgentSolver(env, greedy_agent)
    local_search_solver = SingleAgentSolver(env, local_search_agent)
    sa_solver = SingleAgentSolver(env, simulated_annealing_agent)

    # Crear ILS
    perturbation = Perturbation(env, "flip", pert_size=3)
    ils = ILS(local_search_solver, perturbation, max_iterations=50)

    # Resolver con agente constructivo
    greedy_solution = greedy_solver.solve(initial_state)
    greedy_value = greedy_solution.cut_value

    # Resolver con agente de búsqueda local a partir de la solución constructiva
    local_search_solution = local_search_solver.solve(deepcopy(greedy_solution))
    local_search_value = local_search_solution.cut_value

    # Resolver con ILS a partir de la solución constructiva
    ils_solution = ils.solve(deepcopy(greedy_solution))
    ils_value = ils_solution.cut_value


    sa_solution = sa_solver.solve(deepcopy(greedy_solution))
    sa_value = sa_solution.cut_value

    return {
        "Greedy": greedy_value,
        "Local Search": local_search_value,
        "Simulated Annealing": sa_solution.cut_value,
        "ILS": ils_value
    }

# Experimento con diferentes tamaños de instancia
def run_experiment(
    node_sizes_random=None,          # Lista de tamaños para generar, e.g., [100, 150, 200]
    replicas_per_random_size=5,      # Número de réplicas para cada tamaño aleatorio
    file_paths=None,                 # Lista de rutas a archivos de instancia, e.g., ['g1.rud', 'data/g2.rud']
    random_instance_params=None      # Opcional: dict con params para generate_random_instance (density, weights)
    ):
    """
    Ejecuta un experimento sobre instancias Max-Cut generadas aleatoriamente y/o cargadas desde archivos.

    Args:
        node_sizes_random (list, optional): Lista de N para generar instancias aleatorias. Defaults to None.
        replicas_per_random_size (int, optional): Réplicas por cada tamaño en node_sizes_random. Defaults to 5.
        file_paths (list, optional): Lista de rutas a archivos de instancia para cargar. Defaults to None.
        random_instance_params (dict, optional): Parámetros adicionales para generate_random_instance
                                                  (e.g., {'density': 0.5, 'min_weight': -1, 'max_weight': 1}). Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame con los resultados del experimento.
                          Columnas incluirán info de la instancia y métricas de solve_and_compare.
    """
    results = []
    if random_instance_params is None:
        random_instance_params = {} # Usar defaults de la función generate_random_instance

    # --- Ejecutar sobre Instancias Aleatorias ---
    if node_sizes_random:
        print(f"--- Iniciando Experimentos con Instancias Aleatorias (Tamaños: {node_sizes_random}) ---")
        for n_nodes in node_sizes_random:
            print(f"\nGenerando {replicas_per_random_size} instancias aleatorias de tamaño {n_nodes}...")
            for i in range(replicas_per_random_size):
                print(f"  Réplica {i+1}/{replicas_per_random_size} (n={n_nodes})")
                instance_name = f"random_n{n_nodes}_r{i+1}"
                try:
                    instance = generate_random_instance(n_nodes, **random_instance_params)
                    result = solve_and_compare(instance) # Resuelve esta instancia específica

                    # Añadir información de la instancia al resultado
                    result["instance_name"] = instance_name
                    result["instance_type"] = "random"
                    result["n_nodes"] = n_nodes
                    result["replica"] = i + 1
                    results.append(result)
                except Exception as e:
                    print(f"Error procesando instancia aleatoria {instance_name}: {e}")
        print("--- Fin Experimentos con Instancias Aleatorias ---")


    # --- Ejecutar sobre Instancias de Archivos ---
    if file_paths:
        print(f"\n--- Iniciando Experimentos con Instancias de Archivo (Archivos: {len(file_paths)}) ---")
        for file_path in file_paths:
            instance_name = os.path.basename(file_path) # Obtener nombre base del archivo
            print(f"\nCargando y procesando instancia: {instance_name} ({file_path})")
            if not os.path.exists(file_path):
                print(f"  Error: Archivo no encontrado en {file_path}. Saltando.")
                continue

            try:
                # Usar la función para cargar desde archivo (formato .rud/.G)
                instance = load_rud_file_to_instance(file_path)
                result = solve_and_compare(instance) # Resuelve esta instancia específica

                # Añadir información de la instancia al resultado
                result["instance_name"] = instance_name
                result["instance_type"] = "file"
                result["n_nodes"] = instance.num_vertices # Obtener N desde la instancia cargada
                result["replica"] = 1 # O manejar réplicas si resuelves el mismo archivo varias veces
                results.append(result)

            except (FileNotFoundError, ValueError, Exception) as e:
                print(f"  Error procesando archivo {file_path}: {e}. Saltando.")
        print("--- Fin Experimentos con Instancias de Archivo ---")

    # Convertir la lista de diccionarios de resultados a un DataFrame
    if not results:
         print("\nAdvertencia: No se generaron resultados.")
         return pd.DataFrame() # Retornar DataFrame vacío si no hubo resultados

    print("\nExperimento completado. Creando DataFrame de resultados.")
    return pd.DataFrame(results)


results = run_experiment(file_paths=["./g1/g1.rud"])
results.to_csv("results.csv")