# Estudo de Caso 2 - Minimizando Custos de Consumo de Energia de um Data Center

# Treinamento da Inteligência Artificial

# Importação das bibliotecas
import os
import numpy as np
import random as rn
import environment
import brain
import dqn

# Configuração do seed para reproducibilidade
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# CONFIGURAÇÃO DOS PARÂMETROS
epsilon = .3
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# CRIAÇÃO DO AMBIENTE
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0,
                              initial_number_users = 20, initial_rate_data = 30)

# CRIAÇÃO DO CÉREBRO (REDE NEURAL)
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)

# CRIAÇÃO DO DEEP Q-LEARNING
dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)

# DEFINIÇÃO DO MODO DE TREINAMENTO/TESTE
train = True

# TREINAMENTO DA INTELIGÊNCIA ARTIFICIAL
env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0
if (env.train):
    # COMEÇANDO O LOOP QUE PERCORRERÁ TODAS AS ÉPOCAS (1 ÉPOCA = 5 MESES)
    for epoch in range(1, number_epochs):
        # INICIALIZAÇÃO DAS VARIÁVEIS DO AMBIENTE E DO LOOP
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        
        # COMEÇANDO O LOOP QUE PERCORRERÁ CADA TIMESTEP (1 TIMESTEP = 1 MINUTO)
        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):
            # EXECUÇÃO DA PRÓXIMA AÇÃO COM EXPLORAÇÃO
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            
            # EXECUÇÃO DA PRÓXIMA AÇÃO POR INFERÊNCIA, USANDO A REDE NEURAL
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            
            # ATUALIZAÇÃO DO AMBIENTE E OBTENDO O NOVO ESTADO
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)))
            total_reward += reward
            
            # ARMAZENAMENTO DA NOVA TRANSAÇÃO NA MEMÓRIA
            dqn.remember([current_state, action, reward, next_state], game_over)
            
            # SEPARAÇÃO EM DOIS BATCHES SEPARADOS (INPUTS E TARGETS)
            inputs, targets = dqn.get_batch(model, batch_size = batch_size)
            
            # CÁLCULO DO ERRO (LOSS) USANDO OS BATCHES
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
        
        # IMPRESSÃO DOS RESULTADOS DO TREINAMENTO PARA CADA ÉPOCA
        print('\n')
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total energy spent with AI: {:.0f}".format(env.total_energy_ai))
        print("Total energy spent with no AI: {:.0f}".format(env.total_energy_noai))        
        
        # EARLY STOPPING
        if (early_stopping):
            if (total_reward <= best_total_reward):
                patience_count += 1
            elif (total_reward > best_total_reward):
                best_total_reward = total_reward
                patience_count = 0
            if (patience_count > patience):
                print("Early stopping")
                break
        
        # SALVAR O MODELO
        model.save("model.h5")






































