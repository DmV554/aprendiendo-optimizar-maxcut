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
        'num_episodes': 5000,
        'num_nodes': 30,       # Tamaño de los grafos de entrenamiento
        'density': 0.8,
        'lr': 0.0005,          # Tasa de aprendizaje
        'gamma': 0.99,         # Factor de descuento
        'hidden_dim': 128      # Dimensión de la capa oculta de la GNN
    }
    
    # Dimensiones de las características de entrada para la red
    # [No Asignado, Asignado a P0, Asignado a P1]
    input_dim = 3
    
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