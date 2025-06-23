# train_drl.py (con Curriculum Learning)

import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from MAXCUT import MaxCutInstance, generate_random_instance
from drl_environment import MaxCutDRLEnv
from drl_model import ActorCriticGAT,ActorCriticGAT_V2,ActorCriticGAT_V3 # O ActorCriticGAT_V2, ActorCriticGAT_V3

# --- Hiperparámetros del Algoritmo ---
LEARNING_RATE = 3e-5
GAMMA = 0.99
ENTROPY_BETA = 0.015
VALUE_LOSS_COEFF = 0.6

# --- Estrategia de Curriculum Learning ---
# Formato: (numero_de_episodios_para_la_etapa, tamaño_de_nodos_en_la_etapa)
CURRICULUM = [
    (1500, 20),  # Etapa 1: 1500 episodios con grafos de 20 nodos
    (1500, 30),  # Etapa 2: 1500 episodios con grafos de 30 nodos
    (2000, 40),   # Etapa 3: 2000 episodios con grafos de 40 nodos. Más episodios para la etapa más difícil.
    (2000, 50),
    (3000, 60)
]
DENSITY = 0.8 # Densidad de los grafos generados

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Elige aquí la arquitectura que quieres entrenar con el currículo
model = ActorCriticGAT_V3().to(device) 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Iniciando entrenamiento con Curriculum Learning en: {device}")
print(f"Modelo: {model}")
print(f"Currículo: {CURRICULUM}")

# --- Bucle de Entrenamiento con Currículo ---
episode_rewards = []
global_episode_count = 0

# Bucle externo: itera sobre las etapas del currículo
for num_episodes_stage, num_nodes_stage in CURRICULUM:
    print(f"\n--- Iniciando Etapa del Currículo: {num_episodes_stage} episodios con N={num_nodes_stage} ---")
    
    # Bucle interno: itera sobre los episodios de la etapa actual
    for episode_in_stage in tqdm(range(num_episodes_stage), desc=f"Etapa N={num_nodes_stage}"):
        
        # Generar una nueva instancia con el tamaño de la etapa actual
        weights = generate_random_instance(num_nodes_stage, density=DENSITY)
        instance = MaxCutInstance(weights)
        env = MaxCutDRLEnv(instance)
        
        graph_repr = env.reset()
        
        # Listas para almacenar los resultados del episodio
        log_probs = []
        values = []
        rewards = []
        
        # --- Ejecutar un episodio ---
        done = False
        while not done:
            # Mover datos del grafo al dispositivo
            x, edge_index, edge_weight = graph_repr
            x, edge_index, edge_weight = x.to(device), edge_index.to(device), edge_weight.to(device)

            # Obtener el índice del nodo a decidir
            node_idx = env.get_current_node_to_assign()
            
            # Pasar por el modelo para obtener política y valor
            logits, value = model(x, edge_index, edge_weight, node_idx)
            
            # Crear una distribución de probabilidad y muestrear una acción
            prob_dist = Categorical(logits=logits)
            action = prob_dist.sample() # Muestreo estocástico
            
            # Guardar log-probabilidad y valor
            log_probs.append(prob_dist.log_prob(action))
            values.append(value)
            
            # Ejecutar la acción en el entorno
            graph_repr, reward, done = env.step(action.item())
            rewards.append(reward)

        # --- Fin del episodio: Calcular Pérdida y Actualizar Pesos ---
        
        # 1. Calcular Retornos Descontados (G_t)
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + GAMMA * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Normalizar retornos para mayor estabilidad
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 2. Convertir listas a tensores
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze(-1) # Usar squeeze(-1) para evitar errores si el batch es de 1

        # 3. Calcular Ventaja (Advantage) A_t = G_t - V(s_t)
        advantages = returns - values
        
        # 4. Calcular la Pérdida del Actor (Política)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # 5. Calcular la Pérdida del Crítico (Valor)
        critic_loss = F.smooth_l1_loss(values, returns)
        
        # 6. Calcular la Pérdida de Entropía
        # prob_dist es la del último paso, necesitamos calcular sobre todas las acciones
        entropies = prob_dist.entropy()
        entropy_loss = -entropies.mean()

        # 7. Calcular la Pérdida Total
        loss = actor_loss + VALUE_LOSS_COEFF * critic_loss + ENTROPY_BETA * entropy_loss
        
        # 8. Backpropagation y Optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Logging ---
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        global_episode_count += 1
        
        # Imprimir logs cada 50 episodios globales
        if global_episode_count % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episodio Global {global_episode_count}: Etapa N={num_nodes_stage}, "
                  f"Loss Total={loss.item():.4f}, "
                  f"Recompensa Total={total_reward:.2f}, "
                  f"Recompensa Media (últimos 50)={avg_reward:.2f}")

print("\n--- Entrenamiento Finalizado ---")

# Guardar el modelo entrenado final
torch.save(model.state_dict(), "drl_curriculum_model.pth")
print("Modelo entrenado con currículo guardado en 'drl_curriculum_model.pth'")