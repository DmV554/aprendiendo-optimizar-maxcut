# src/run_drl.py

from DRLTrainer import DRLTrainer
from DRLAgent import A2CAgent
import torch

def main():
    """
    Punto de entrada para ejecutar el entrenamiento del agente DRL.
    """
    # Configuración del entrenamiento
    config = {
        'num_episodes': 20000,
        'num_nodes': 30,
        'density': 0.8,
        'lr': 0.0001,  # Estaba en 0.0005
        'gamma': 0.99,
        'hidden_dim': 128,
        'entropy_coef': 0.01,
        'ordering_strategy': 'weighted_degree_high_low'  # Estrategia de ordenamiento de nodos
    }
    # Dimensiones de las características de entrada para la red
    input_dim = 8
    
    # Inicializar el agente
    agent = A2CAgent(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        gamma=config['gamma']
    )
    
    # Inicializar y ejecutar el entrenador
    trainer = DRLTrainer(agent, config)
    trainer.train()
    ben = "hola"
    print(ben)

    # Opcional: Guardar el modelo entrenado
    torch.save(agent.policy_network.state_dict(), 'drl_maxcut_model.pth')
    print("Modelo guardado en 'drl_maxcut_model.pth'")

if __name__ == '__main__':
    main()