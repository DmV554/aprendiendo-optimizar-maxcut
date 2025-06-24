# compare_agents.py (Versión Refinada)

import os
import json
import torch
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from copy import deepcopy
from datetime import datetime

# --- Importar Clases y Funciones Necesarias ---
from MAXCUT import MaxCutInstance, MaxCutState, MaxCutEnvironment, generate_random_instance
from Agents import GreedyAgent, FirstImprovementAgent, SingleAgentSolver, Perturbation, ILS, evalConstructiveActions
from drl_agent import DRLAgent

# =============================================================================
# --- CONFIGURACIÓN CENTRAL DEL EXPERIMENTO ---
# =============================================================================

EXPERIMENT_NAME = "Hybrid_vs_ILS_Final"
MODEL_PATH = "drl_curriculum_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENERAL_CONFIG = {
    "instance_sizes": [20, 40, 60, 80, 100],
    "num_instances_per_size": 10,
    "density": 0.8,
    "min_weight": 1,
    "max_weight": 100,
}

# --- Diccionario de Estrategias a Probar ---
# Aquí defines qué agentes quieres comparar. Añade o quita entradas según necesites.
STRATEGIES_TO_RUN = {
    "drl_g": {
        "name": "DRL-G (Puro)",
        "type": "constructive",
        "agent_class": DRLAgent,
        "agent_params": {"model_path": MODEL_PATH, "device": DEVICE},
        "plot_style": {"marker": "o", "linestyle": "-"}
    },
    "heur_g": {
        "name": "Heur-G (Baseline)",
        "type": "constructive",
        "agent_class": GreedyAgent,
        "agent_params": {"eval_actions": evalConstructiveActions},
        "plot_style": {"marker": "s", "linestyle": "-"}
    },
    "hybrid_drl_ls": {
        "name": "DRL-G + LS (Híbrido)",
        "type": "hybrid",
        "constructor_agent_key": "drl_g", # Clave del agente constructor a usar
        "local_search_params": {
            "neighborhood_type": "pair_swap",
            "max_actions": 1500
        },
        "plot_style": {"marker": "*", "linestyle": "-", "markersize": 12, "linewidth": 2.5}
    },
    "ils": {
        "name": "ILS (Experto)",
        "type": "ils",
        "local_search_params": { # Parámetros para la búsqueda local dentro de ILS
            "neighborhood_type": "pair_swap",
            "max_actions": 1500
        },
        "ils_params": { # Parámetros específicos de ILS
            "pert_size_factor": 0.20,
            "max_iterations": 150
        },
        "plot_style": {"marker": "P", "linestyle": "--"}
    }
}

# =============================================================================
# --- LÓGICA PRINCIPAL DEL SCRIPT ---
# =============================================================================

def main():
    # --- 1. Preparar Entorno y Directorios de Salida ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join("results", f"{EXPERIMENT_NAME}_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)
    
    print(f"Iniciando comparación: {EXPERIMENT_NAME}")
    print(f"Los resultados se guardarán en: {results_folder}")
    print(f"Probando tamaños de instancia: {GENERAL_CONFIG['instance_sizes']}")
    print(f"Estrategias a ejecutar: {list(STRATEGIES_TO_RUN.keys())}")
    print("---")

    # --- 2. Inicializar Agentes y Solvers reutilizables ---
    agents = {}
    for key, config in STRATEGIES_TO_RUN.items():
        if config['type'] == 'constructive':
            agents[key] = config['agent_class'](**config['agent_params'])
    
    # --- 3. Bucle de Comparación ---
    results_data = []
    for n_nodes in GENERAL_CONFIG["instance_sizes"]:
        print(f"\n--- Evaluando tamaño N = {n_nodes} ---")
        
        for i in tqdm(range(GENERAL_CONFIG["num_instances_per_size"]), desc=f"Instancias N={n_nodes}"):
            instance_weights = generate_random_instance(n_nodes, density=GENERAL_CONFIG["density"], min_weight=GENERAL_CONFIG["min_weight"], max_weight=GENERAL_CONFIG["max_weight"])
            instance = MaxCutInstance(instance_weights)
            instance_results = {'num_nodes': n_nodes, 'instance_id': i}

            # Ejecutar cada estrategia definida en la configuración
            for key, config in STRATEGIES_TO_RUN.items():
                start_time = time.time()
                final_cut = -1

                if config['type'] == 'constructive':
                    agent = agents[key]
                    if hasattr(agent, 'reset'): agent.reset()
                    solver = SingleAgentSolver(MaxCutEnvironment(), agent)
                    final_state = solver.solve(MaxCutState(instance))
                    final_cut = final_state.cut_value
                
                elif config['type'] == 'hybrid':
                    constructor_key = config['constructor_agent_key']
                    constructor_agent = agents[constructor_key]
                    if hasattr(constructor_agent, 'reset'): constructor_agent.reset()
                    
                    # Paso A: Construcción
                    constructor_solver = SingleAgentSolver(MaxCutEnvironment(), constructor_agent)
                    initial_solution = constructor_solver.solve(MaxCutState(instance))
                    
                    # Paso B: Refinamiento
                    ls_params = config['local_search_params']
                    ls_agent = FirstImprovementAgent(neighborhood_type=ls_params['neighborhood_type'])
                    ls_solver = SingleAgentSolver(MaxCutEnvironment(), ls_agent, max_actions=ls_params['max_actions'])
                    final_state = ls_solver.solve(deepcopy(initial_solution))
                    final_cut = final_state.cut_value

                elif config['type'] == 'ils':
                    ls_params = config['local_search_params']
                    ils_params = config['ils_params']
                    
                    ls_agent = FirstImprovementAgent(neighborhood_type=ls_params['neighborhood_type'])
                    ls_solver = SingleAgentSolver(MaxCutEnvironment(), ls_agent, max_actions=ls_params['max_actions'])
                    
                    pert_size = max(4, int(n_nodes * ils_params['pert_size_factor']))
                    perturbation = Perturbation(MaxCutEnvironment(), type=ls_params['neighborhood_type'], pert_size=pert_size)
                    
                    ils_solver = ILS(local_search=ls_solver, perturbation=perturbation, max_iterations=ils_params['max_iterations'])
                    
                    initial_partition = [random.choice([0, 1]) for _ in range(n_nodes)]
                    final_state = ils_solver.solve(MaxCutState(instance, partition=initial_partition))
                    final_cut = final_state.cut_value

                instance_results[f'{key}_time'] = time.time() - start_time
                instance_results[f'{key}_cut'] = final_cut
                
            results_data.append(instance_results)

    # --- 4. Procesamiento y Guardado de Resultados ---
    if not results_data:
        print("No se generaron resultados.")
        return

    df_results = pd.DataFrame(results_data)
    
    # Calcular Gaps dinámicamente
    expert_key = 'ils' # Definir quién es el benchmark
    for key in STRATEGIES_TO_RUN:
        if key != expert_key:
            df_results[f'{key}_gap'] = (df_results[f'{expert_key}_cut'] - df_results[f'{key}_cut']) / abs(df_results[f'{expert_key}_cut']) * 100

    # Guardar resultados completos en CSV
    df_results.to_csv(os.path.join(results_folder, "full_results.csv"), index=False)
    
    # Calcular y guardar resumen promedio
    avg_results = df_results.groupby('num_nodes').mean()
    summary_path = os.path.join(results_folder, "summary.json")
    avg_results.to_json(summary_path, orient='index', indent=4)
    print(f"\n\n--- Resumen Promedio (guardado en {summary_path}) ---")
    print(avg_results)
    
    # --- 5. Graficación Dinámica ---
    print("\n--- Generando Gráficos ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    palette = sns.color_palette("viridis", n_colors=len(STRATEGIES_TO_RUN))
    strategy_colors = {key: color for key, color in zip(STRATEGIES_TO_RUN.keys(), palette)}

    plot_configs = {
        "quality": {"y": "cut", "title": "Comparación de Calidad de Solución (Valor del Corte)", "ylabel": "Valor Promedio del Corte"},
        "time": {"y": "time", "title": "Comparación de Tiempos de Ejecución", "ylabel": "Tiempo Promedio de Ejecución (segundos)"},
        "gap": {"y": "gap", "title": "Gap de Solución Promedio vs. ILS", "ylabel": "Gap Promedio (%) [Menor es Mejor]"}
    }

    for plot_key, p_config in plot_configs.items():
        plt.figure(figsize=(12, 7))
        for key, config in STRATEGIES_TO_RUN.items():
            if f'{key}_{p_config["y"]}' in df_results.columns:
                plot_params = config['plot_style']
                plot_params['color'] = strategy_colors[key]
                sns.lineplot(data=df_results, x='num_nodes', y=f'{key}_{p_config["y"]}', label=config['name'], **plot_params)
        
        plt.title(p_config["title"])
        plt.xlabel('Número de Nodos (N)')
        plt.ylabel(p_config["ylabel"])
        if plot_key == 'time': plt.yscale('log')
        if plot_key == 'gap': plt.axhline(0, color='grey', linestyle='--', label=f'Nivel {STRATEGIES_TO_RUN[expert_key]["name"]}')
        
        plt.xticks(GENERAL_CONFIG["instance_sizes"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f"plot_{plot_key}.png"))
        plt.show()

    print(f"\nComparación finalizada. Gráficos guardados en la carpeta: {results_folder}")


if __name__ == "__main__":
    main()