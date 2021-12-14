# -*- coding: utf-8 -*-
"""Minimizando Custos no Consumo de Energia de um Data Center com o uso do agente Deep Q-Learning
"""

# Luis Magno da Silva Novaes
# Construindo o Ambiente
# Importando as bibliotecas
import numpy as np

# 1 - CONSTRUINDO O AMBIENTE EM UMA CLASSE

class Environment(object):

# 2 - INTRODUZIR E INICIALIZAR TODOS OS PARÂMETROS E VARIÁVEIS DO AMBIENTE

  def __init__(self,
              optimal_temperature = (18.0, 24.0),
              initial_month = 0,
              initial_number_users = 10,
              initial_rate_data = 60):
      self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0,
                                              23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
      self.initial_month = initial_month
      self.atmospheric_temperature = \
                              self.monthly_atmospheric_temperatures[initial_month]
      self.optimal_temperature = optimal_temperature
      self.min_temperature = -20
      self.max_temperature = 80
      self.min_number_users = 10
      self.max_number_users = 100
      self.max_update_users = 5
      self.min_rate_data = 20
      self.max_rate_data = 300
      self.max_update_data = 10
      self.initial_number_users = initial_number_users
      self.current_number_users = initial_number_users
      self.initial_rate_data = initial_rate_data
      self.current_rate_data = initial_rate_data
      self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
      self.temperature_ai = self.intrinsic_temperature
      self.temperature_noai = (self.optimal_temperature[0]
                                   + self.optimal_temperature[1]) / 2.0
      self.total_energy_ai = 0.0
      self.total_energy_noai = 0.0
      self.reward = 0.0
      self.game_over = 0
      self.train = 1

# 3 - CONSTRIR UM MÉTODO QUE ATUALIZA O AMBIENTE APÓS O AGENTE EXECUTAR UMA AÇÃO

  def update_env(self, direction, energy_ai, month):

# 4 - OBTER A RECOMPENSA

      # Calcular a energia gasta pelo sistema de resfriamento do servidor quando não há Agente
      energy_noai = 0
      if (self.temperature_noai < self.optimal_temperature[0]):
          energy_noai = self.optimal_temperature[0] - self.temperature_noai
          self.temperature_noai = self.optimal_temperature[0]
      elif (self.temperature_noai > self.optimal_temperature[1]):
          energy_noai = self.temperature_noai - self.optimal_temperature[1]
          self.temperature_noai = self.optimal_temperature[1]
      # Calcular a recompensa
      self.reward = energy_noai - energy_ai
      # Escalar a recompensa
      self.reward = 1e-3 * self.reward

      
# 5 - OBTER O PRÓXIMO ESTADO

      # Atualizar a temperatura atmosférica
      self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]
      # Atualizando o número de usuários
      self.current_number_users += np.random.randint(-self.max_update_users,
                                                    self.max_update_users)
      if (self.current_number_users > self.max_number_users):
          self.current_number_users = self.max_number_users
      elif (self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users
      # Atualizar a taxa de dados
      self.current_rate_data += np.random.randint(-self.max_update_data,
                                                    self.max_update_data)
      if (self.current_rate_data > self.max_rate_data):
          self.current_rate_data = self.max_rate_data
      elif (self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
      # Calcular o Delta da Temperatura Intrínseca
      past_intrinsic_temperature = self.intrinsic_temperature
      self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
      delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature
      # Calcular o Delta de Temperatura causado pela Agente
      if (direction == -1):
          delta_temperature_ai = -energy_ai
      elif (direction == 1):
            delta_temperature_ai = energy_ai
      # Atualizar a temperatura do novo servidor quando houver Agente
      self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai
      # Atualizar a temperatura do novo servidor quando não houver Agente
      self.temperature_noai += delta_intrinsic_temperature

# 6 - FINALIZANDO A EXECUÇÃO

      if (self.temperature_ai < self.min_temperature):
          if (self.train == 1):
              self.game_over = 1
          else:
              self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
              self.temperature_ai = self.optimal_temperature[0]
      elif (self.temperature_ai > self.max_temperature):
          if (self.train == 1):
              self.game_over = 1
          else:
              self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
              self.temperature_ai = self.optimal_temperature[1]


# 7 - ATUALIZAR A PONTUAÇÃO

      # Atualizar a energia total gasta pelo Agente
      self.total_energy_ai += energy_ai
      # Atualizar a Energia Total gasta pelo sistema alternativo quando não houver Agente
      self.total_energy_noai += energy_noai

# 8 - ESCALANDO O PRÓXIMO ESTADO

      scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
      scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
      scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
      next_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
      
# 9 - RETORNAR O PRÓXIMO ESTADO, A RECOMPENSA E A EXECUÇÃO
      
      return next_state, self.reward, self.game_over

# 10 - CONSTRUIR UM MÉTODO QUE REINICIALIZA O AMBIENTE

  def reset(self, new_month):
      self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
      self.initial_month = new_month
      self.current_number_users = self.initial_number_users
      self.current_rate_data = self.initial_rate_data
      self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
      self.temperature_ai = self.intrinsic_temperature
      self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
      self.total_energy_ai = 0.0
      self.total_energy_noai = 0.0
      self.reward = 0.0
      self.game_over = 0
      self.train = 1

# 11 - CONSTRUIR UM MÉTODO QUE TRAGA A QUALQUER MOMENTO O ESTADO, A RECOMPENSA E O GAMEOVER

  def observe(self):
      scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
      scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
      scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
      current_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
      return current_state, self.reward, self.game_over

"""# ETAPA - 2"""

# Construindo o cérebro
# Importando as bibliotecas
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam

# CONSTRUINDO O CÉREBRO
class Brain(object):

    # CONSTRUINDO UMA REDE NEURAL TOTALMENTE CONECTADA DIRETAMENTE DENTRO DO MÉTODO INIT
    def __init__(self, learning_rate = 0.001, number_actions = 5):
      self.learning_rate = learning_rate

    # CONSTRUINDO A CAMADA DE ENTRADA COMPOSTA PELO ESTADO DE ENTRADA
      states = Input(shape = (3,))


    # CONSTRUINDO A PRIMEIRA CAMADA OCULTA TOTALMENTE CONECTADA COM DROPOUT ATIVADO
      x = Dense(units = 64, activation = 'sigmoid')(states)
      x = Dropout(rate = 0.1)(x)

    # CONSTRUINDO A SEGUNDA CAMADA OCULTA TOTALMENTE CONECTADA COM DROPOUT ATIVADO
      y = Dense(units = 32, activation = 'sigmoid')(x)
      y = Dropout(rate = 0.1)(y)

    # CONSTRUINDO A CAMADA DE SAÍDA, TOTALMENTE CONECTADA À ÚLTIMA CAMADA OCULTA
      q_values = Dense(units = number_actions, activation = 'softmax')(y)

    # MONTAGEM DA ARQUITETURA COMPLETA DENTRO DE UM MODELO DE OBJETO
      self.model = Model(inputs = states, outputs = q_values)

    # COMPILANDO O MODELO COM UMA PERDA DE ERRO QUADRADO MÉDIO E UM OTIMIZADOR ESCOLHIDO
      self.model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))

"""# ETAPA - 3"""

# Implementar Deep Q-Learning com Experience Replay
# Importar as bibliotecas
import numpy as np

# IMPLEMENTAÇÃO DE APRENDIZAGEM DE Q PROFUNDA COM REPLAY DE EXPERIÊNCIA
class DQN(object):

      # INTRODUZIR E INICIALIZAR TODOS OS PARÂMETROS E VARIÁVEIS DO DQN
      def __init__(self, max_memory = 100, discount = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount

      # CONTRUIR A MEMÓRIA NA REPETIÇÃO DE EXPERIÊNCIA
      def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

      # CONSTRUIR DOIS LOTES DE ENTRADAS E ALVOS
      def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]
        num_outputs = model.output_shape[-1]
        inputs = np.zeros((min(len_memory, batch_size), num_inputs))
        targets = np.zeros((min(len_memory, batch_size), num_outputs))

        for i, idx in enumerate(np.random.randint(0, len_memory,
          size = min(len_memory, batch_size))):
          current_state, action, reward, next_state = self.memory[idx][0]
          game_over = self.memory[idx][1]
          inputs[i] = current_state
          targets[i] = model.predict(current_state)[0]
          Q_sa = np.max(model.predict(next_state)[0])
          if game_over:
            targets[i, action] = reward
          else:
            targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets

"""# ETAPA - 4 """

# Training the AI
# Installing Keras
# conda install -c conda-forge keras
# Importing the libraries and the other python files
#!pip install environment
#!pip install brain
#!pip install dqn
#!pip install stringIO

import os
import numpy as np
import random as rn
import environment
import brain
import dqn

# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING THE PARAMETERS
epsilon = .3
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# BUILDING THE ENVIRONMENT BY SIMPLY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = environment.Environment(optimal_temperature = (18.0, 24.0),
initial_month = 0,
initial_number_users = 20,
initial_rate_data = 30)

# BUILDING THE BRAIN BY SIMPLY CREATING AN OBJECT OF THE BRAIN CLASS
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)

# BUILDING THE DQN MODEL BY SIMPLY CREATING AN OBJECT OF THE DQN CLASS
dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)

# CHOOSING THE MODE
train = True

# TRAINING THE AI
env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0
if (env.train):

    # STARTING THE LOOP OVER ALL THE EPOCHS (1 Epoch = 5 Months)
    for epoch in range(1, number_epochs):

        # INITIALIAZING ALL THE VARIABLES OF BOTH THE ENVIRONMENT AND THE TRAINING LOOP
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0

        # STARTING THE LOOP OVER ALL THE TIMESTEPS (1 Timestep = 1 Minute) IN ONE EPOCH
        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):

            # PLAYING THE NEXT ACTION BY EXPLORATION
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                        direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step

            # PLAYING THE NEXT ACTION BY INFERENCE
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step

            # UPDATING THE ENVIRONMENT AND REACHING THE NEXT STATE
            next_state, reward, game_over = env.update_env(direction,
            energy_ai,
            int(timestep / (30*24*60)))
            total_reward += reward

            # STORING THIS NEW TRANSITION INTO THE MEMORY
            dqn.remember([current_state, action, reward, next_state], game_over)

            # GATHERING IN TWO SEPARATE BATCHES THE INPUTS AND THE TARGETS
            inputs, targets = dqn.get_batch(model, batch_size = batch_size)

            # COMPUTING THE LOSS OVER THE TWO WHOLE BATCHES OF INPUTS AND TARGETS
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state

        # PRINTING THE TRAINING RESULTS FOR EACH EPOCH
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))

        # EARLY STOPPING
        if (early_stopping):
              if (total_reward <= best_total_reward):
                  patience_count += 1
              elif (total_reward > best_total_reward):
                  best_total_reward = total_reward
                  patience_count = 0
              if (patience_count >= patience):
                print("Early Stopping")
                break

        # SAVING THE MODEL
        model.save("model.h5")
