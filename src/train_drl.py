# train_drl.py

import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from MAXCUT import MaxCutInstance, generate_random_instance
from drl_environment import MaxCutDRLEnv
from drl_model import ActorCriticGAT

# --- Hiperparámetros ---
LEARNING_RATE = 1e-4
GAMMA = 0.99  # Factor de descuento para recompensas futuras
ENTROPY_BETA = 0.01 # Coeficiente para el bono de entropía
VALUE_LOSS_COEFF = 0.5 # Coeficiente para la pérdida del crítico

NUM_EPISODES = 2000
NUM_NODES = 30 
DENSITY = 0.8

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCriticGAT().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Iniciando entrenamiento en: {device}")
print(f"Modelo: {model}")

# --- Bucle de Entrenamiento ---
episode_rewards = []

for episode in tqdm(range(NUM_EPISODES)):
    # Generar una nueva instancia de grafo para cada episodio para mejorar la generalización
    weights = generate_random_instance(NUM_NODES, density=DENSITY)
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
    values = torch.stack(values).squeeze()

    # 3. Calcular Ventaja (Advantage) A_t = G_t - V(s_t)
    advantages = returns - values
    
    # 4. Calcular la Pérdida del Actor (Política)
    # El .detach() en advantages evita que los gradientes fluyan hacia el crítico desde aquí
    actor_loss = -(log_probs * advantages.detach()).mean()

    # 5. Calcular la Pérdida del Crítico (Valor)
    # Compara el valor predicho 'values' con los retornos reales 'returns'
    critic_loss = F.smooth_l1_loss(values, returns)
    
    # 6. Calcular la Pérdida de Entropía
    # Queremos maximizar la entropía, así que minimizamos su negación
    entropy_loss = -prob_dist.entropy().mean()

    # 7. Calcular la Pérdida Total
    loss = actor_loss + VALUE_LOSS_COEFF * critic_loss + ENTROPY_BETA * entropy_loss
    
    # 8. Backpropagation y Optimización
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # --- Logging ---
    total_reward = sum(rewards)
    episode_rewards.append(total_reward)
    if episode % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episodio {episode}: Loss Total={loss.item():.4f}, "
              f"Recompensa Total={total_reward:.2f}, "
              f"Recompensa Media (últimos 50)={avg_reward:.2f}")

print("\n--- Entrenamiento Finalizado ---")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "drl_actor_critic_model.pth")
print("Modelo guardado en 'drl_actor_critic_model.pth'")