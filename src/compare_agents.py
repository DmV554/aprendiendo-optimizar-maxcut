# compare_agents.py

import torch
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- Importar Clases y Funciones Necesarias ---
# Asegúrate de que todos estos archivos estén accesibles
from MAXCUT import MaxCutInstance, MaxCutState, MaxCutEnvironment, generate_random_instance
from Agents import GreedyAgent, FirstImprovementAgent, SingleAgentSolver, Perturbation, ILS, evalConstructiveActions
from drl_agent import DRLAgent

# --- Configuración del Experimento ---
MODEL_PATH = "drl_actor_critic_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INSTANCE_SIZES = [20, 30, 40, 50, 60] # Tamaños de grafos a probar
NUM_INSTANCES_PER_SIZE = 10 # Número de instancias aleatorias por cada tamaño
DENSITY = 0.8
MIN_WEIGHT = 1
MAX_WEIGHT = 100

print(f"Iniciando comparación en dispositivo: {DEVICE}")
print(f"Probando tamaños de instancia: {INSTANCE_SIZES}")
print(f"Número de instancias por tamaño: {NUM_INSTANCES_PER_SIZE}")
print("---")


# --- Inicialización de Agentes ---

# 1. Agente DRL
try:
    drl_agent = DRLAgent(MODEL_PATH, device=DEVICE)
    print("Agente DRL cargado exitosamente.")
except FileNotFoundError:
    print(f"ERROR: No se encontró el modelo entrenado en '{MODEL_PATH}'.")
    print("Por favor, asegúrate de que el archivo existe o entrena el modelo primero.")
    exit()

# 2. Agente Heurístico Greedy
heuristic_greedy_agent = GreedyAgent(eval_actions=evalConstructiveActions)

# 3. Solver para los agentes constructivos (DRL y Heurístico)
# Un solver simple que llama a la política del agente hasta que la solución es completa
constructive_solver_drl = SingleAgentSolver(MaxCutEnvironment(), drl_agent)
constructive_solver_heuristic = SingleAgentSolver(MaxCutEnvironment(), heuristic_greedy_agent)

# 4. Agente Experto (ILS) como Benchmark
# Configuración de ILS (puedes ajustarla según tus experimentos anteriores)
ils_ls_agent = FirstImprovementAgent(neighborhood_type="flip")
ils_ls_solver = SingleAgentSolver(MaxCutEnvironment(), ils_ls_agent, max_actions=1500)
ils_pert_size_factor = 0.20 # 20% de los nodos
ils_pert = lambda n: Perturbation(MaxCutEnvironment(), type="flip", pert_size=max(4, int(n * ils_pert_size_factor)))
ils_solver_gen = lambda n: ILS(local_search=ils_ls_solver, perturbation=ils_pert(n), max_iterations=150)


# --- Bucle de Comparación ---
results_data = []

for n_nodes in INSTANCE_SIZES:
    print(f"\n--- Evaluando tamaño N = {n_nodes} ---")
    
    # Se genera un solver ILS específico para el tamaño del grafo
    ils_solver = ils_solver_gen(n_nodes)
    
    for i in tqdm(range(NUM_INSTANCES_PER_SIZE), desc=f"Instancias N={n_nodes}"):
        instance_weights = generate_random_instance(n_nodes, density=DENSITY, min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT)
        instance = MaxCutInstance(instance_weights)
        
        instance_results = {'num_nodes': n_nodes, 'instance_id': i}

        # --- 1. Evaluar Agente DRL (DRL-G) ---
        drl_agent.reset()  # <--- AÑADIR ESTA LÍNEA AQUÍ
        state_drl = MaxCutState(instance)
        start_time = time.time()
        final_state_drl = constructive_solver_drl.solve(state_drl)
        instance_results['drl_g_time'] = time.time() - start_time
        instance_results['drl_g_cut'] = final_state_drl.cut_value

        # --- 2. Evaluar Agente Heurístico (Heur-G) ---
        state_heuristic = MaxCutState(instance)
        start_time = time.time()
        final_state_heuristic = constructive_solver_heuristic.solve(state_heuristic)
        instance_results['heur_g_time'] = time.time() - start_time
        instance_results['heur_g_cut'] = final_state_heuristic.cut_value

        # --- 3. Evaluar Agente Experto (ILS) ---
        initial_partition_ils = [random.choice([0, 1]) for _ in range(n_nodes)]
        state_ils = MaxCutState(instance, partition=initial_partition_ils)
        start_time = time.time()
        final_state_ils = ils_solver.solve(state_ils)
        instance_results['ils_time'] = time.time() - start_time
        instance_results['ils_cut'] = final_state_ils.cut_value

        results_data.append(instance_results)

# --- Procesamiento y Visualización de Resultados ---
if not results_data:
    print("No se generaron resultados.")
    exit()

df_results = pd.DataFrame(results_data)

# Calcular el "Gap" de la solución respecto a ILS (cuánto peor es, en porcentaje)
# Gap = (Cut_ILS - Cut_Agente) / |Cut_ILS|
# Un gap más pequeño es mejor.
df_results['drl_gap'] = (df_results['ils_cut'] - df_results['drl_g_cut']) / abs(df_results['ils_cut']) * 100
df_results['heur_gap'] = (df_results['ils_cut'] - df_results['heur_g_cut']) / abs(df_results['ils_cut']) * 100


print("\n\n--- Resultados Promedio por Tamaño de Instancia ---")
avg_results = df_results.groupby('num_nodes').mean()
print(avg_results[['drl_g_cut', 'heur_g_cut', 'ils_cut', 'drl_gap', 'heur_gap']].round(2))
print("\n--- Tiempos Promedio (segundos) ---")
print(avg_results[['drl_g_time', 'heur_g_time', 'ils_time']].round(4))

# --- Graficación ---
plt.style.use('seaborn-v0_8-whitegrid')

# Gráfico 1: Calidad de la Solución (Valor del Corte)
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_results, x='num_nodes', y='drl_g_cut', label='DRL-G (Propuesto)', marker='o', errorbar='sd')
sns.lineplot(data=df_results, x='num_nodes', y='heur_g_cut', label='Heur-G (Baseline)', marker='s', errorbar='sd')
sns.lineplot(data=df_results, x='num_nodes', y='ils_cut', label='ILS (Experto)', marker='P', linestyle='--', errorbar='sd')
plt.title('Comparación de Calidad de Solución (Valor del Corte)')
plt.xlabel('Número de Nodos (N)')
plt.ylabel('Valor Promedio del Corte')
plt.xticks(INSTANCE_SIZES)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_plot_quality.png")
plt.show()


# Gráfico 2: Tiempos de Ejecución
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_results, x='num_nodes', y='drl_g_time', label='DRL-G (Propuesto)', marker='o', errorbar='sd')
sns.lineplot(data=df_results, x='num_nodes', y='heur_g_time', label='Heur-G (Baseline)', marker='s', errorbar='sd')
sns.lineplot(data=df_results, x='num_nodes', y='ils_time', label='ILS (Experto)', marker='P', linestyle='--', errorbar='sd')
plt.title('Comparación de Tiempos de Ejecución')
plt.xlabel('Número de Nodos (N)')
plt.ylabel('Tiempo Promedio de Ejecución (segundos)')
plt.yscale('log') # Escala logarítmica es útil para ver grandes diferencias de tiempo
plt.xticks(INSTANCE_SIZES)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_plot_time.png")
plt.show()

# Gráfico 3: Gap de Optimalidad (%) respecto a ILS
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_results, x='num_nodes', y='drl_gap', label='Gap DRL vs ILS', marker='o', errorbar='sd')
sns.lineplot(data=df_results, x='num_nodes', y='heur_gap', label='Gap Heur-G vs ILS', marker='s', errorbar='sd')
plt.title('Gap de Solución Promedio vs. ILS')
plt.xlabel('Número de Nodos (N)')
plt.ylabel('Gap Promedio (%) [Menor es Mejor]')
plt.axhline(0, color='grey', linestyle='--', label='Nivel de ILS')
plt.xticks(INSTANCE_SIZES)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_plot_gap.png")
plt.show()

print("\nComparación finalizada. Gráficos guardados como 'comparison_plot_*.png'")