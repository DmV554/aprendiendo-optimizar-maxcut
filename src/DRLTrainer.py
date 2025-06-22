# src/DRLTrainer.py

import torch
from tqdm import tqdm
from MAXCUT import generate_random_instance, MaxCutInstance
from DRLEnvironment import DRLEnvironment
from DRLAgent import A2CAgent

class DRLTrainer:
    """
    Orquesta el proceso de entrenamiento del agente DRL para Max-Cut.
    """
    def __init__(self, agent: A2CAgent, config: dict):
        """
        Inicializa el entrenador.

        Args:
            agent (A2CAgent): El agente a entrenar.
            config (dict): Diccionario de configuración para el entrenamiento.
        """
        self.agent = agent
        self.config = config
        self.losses = []
        self.all_time_best_cut = -float('inf')

    def train(self):
        """
        Ejecuta el bucle de entrenamiento principal.
        """
        num_episodes = self.config.get('num_episodes', 1000)
        num_nodes = self.config.get('num_nodes', 20)
        density = self.config.get('density', 0.8)
        
        progress_bar = tqdm(range(num_episodes), desc="Entrenamiento DRL")

        for episode in progress_bar:
            # Generar una nueva instancia para cada episodio para mejorar la generalización
            weights = generate_random_instance(num_nodes, density=density, min_weight=10, max_weight=20)
            instance = MaxCutInstance(weights)
            env = DRLEnvironment(instance)
            
            state = env.reset()
            done = False
            total_episode_reward = 0
            
           

            # Un episodio consiste en construir una solución completa 
            while not done:
                current_node_idx = env.node_order[env.current_step].item()
                action = self.agent.select_action(state, current_node_idx)
                next_state, reward, done = env.step(action)
                
                self.agent.store_outcome(reward, done)
                total_episode_reward += reward
                state = next_state

            # Actualizar la política al final del episodio
            loss = self.agent.update_policy(state)
            self.losses.append(loss)

            # Evaluar y registrar
            final_cut = env.get_final_cut_value()
            if final_cut > self.all_time_best_cut:
                self.all_time_best_cut = final_cut
            
            if episode % 50 == 0:
                progress_bar.set_postfix({
                    'Pérdida': f'{loss:.3f}',
                    # --- CAMBIO AQUÍ: Usa total_episode_reward ---
                    'Corte Episodio': f'{total_episode_reward:.2f}',
                    'Mejor Corte': f'{self.all_time_best_cut:.2f}'
                })
                 
        print("Entrenamiento finalizado.")
        print(f"Mejor valor de corte encontrado durante el entrenamiento: {self.all_time_best_cut}")