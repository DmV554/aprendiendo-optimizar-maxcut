# src/evaluate_drl.py

import torch
from DRLAgent import A2CAgent
from DRLEnvironment import DRLEnvironment
from MAXCUT import generate_random_instance, MaxCutInstance

def evaluate_agent(config, model_path, num_test_graphs=10):
    """
    Carga un agente entrenado y lo evalúa en un conjunto de grafos de prueba.
    """
    print("Iniciando evaluación...")
    
    input_dim = 3
    
    # 1. Cargar el agente y el modelo entrenado
    agent = A2CAgent(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim']
    )
    agent.policy_network.load_state_dict(torch.load(model_path))
    print(f"Modelo cargado desde '{model_path}'")

    total_cut_value = 0
    
    # 2. Bucle de evaluación en múltiples grafos
    for i in range(num_test_graphs):
        # Generar una nueva instancia de prueba
        weights = generate_random_instance(config['num_nodes'], density=config['density'], min_weight=10, max_weight=20)
        instance = MaxCutInstance(weights)
        env = DRLEnvironment(instance)
        
        state = env.reset(shuffle_nodes=False) # No es necesario el shuffle en la evaluación
        done = False
        episode_rewards = []
        
        # 3. Construir la solución de forma determinista
        while not done:
            current_node_idx = env.node_order[env.current_step].item()
            
            agent.policy_network.eval()
            with torch.no_grad():
                action_probs, _ = agent.policy_network(state, current_node_idx)
                action = torch.argmax(action_probs).item()

                if i == 0 and env.current_step < 5:
                    print(f"  Paso {env.current_step}: Probs={action_probs.numpy()}, Acción={action}")

            # Usar el método greedy para seleccionar la acción
            action = agent.select_action_greedy(state, current_node_idx)
            next_state, reward, done = env.step(action)
            episode_rewards.append(reward)
            state = next_state
            
        # 4. Obtener y registrar el resultado
        final_cut = env.get_final_cut_value()
        # Verificación: la suma de recompensas debe ser igual al corte final
        assert abs(final_cut - sum(episode_rewards)) < 1e-5
        
        print(f"Grafo de prueba {i+1}/{num_test_graphs}: Valor del corte = {final_cut:.2f}")
        total_cut_value += final_cut
        
    # 5. Calcular y mostrar métricas finales
    avg_cut_value = total_cut_value / num_test_graphs
    print("\n--- Resultados de la Evaluación ---")
    print(f"Número de grafos de prueba: {num_test_graphs}")
    print(f"Valor promedio del corte: {avg_cut_value:.2f}")
    print("---------------------------------")

if __name__ == '__main__':
    # Usa la misma configuración que en el entrenamiento o una específica para la prueba
    config = {
        'num_nodes': 30,
        'density': 0.8,
        'hidden_dim': 128
    }
    model_path = 'drl_maxcut_model.pth'
    evaluate_agent(config, model_path, num_test_graphs=50)