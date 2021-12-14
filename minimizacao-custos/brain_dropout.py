# Estudo de Caso 2 - Minimizando Custos de Consumo de Energia de um Data Center

# Importação das bibliotecas
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

# CONSTRUÇÃO DO CÉREBRO (REDE NEURAL)

class Brain(object):
    
    # CONSTRUÇÃO DA ARQUITETURA DA REDE NEURAL DENSA DENTRO DO MÉTODO INIT
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate
        
        # CRIAÇÃO DA CAMADA DE ENTRADA COMPOSTA PELO INPUT STATE
        states = Input(shape = (3,))
        
        # CRIAÇÃO DA PRIMEIRA CAMADA OCULTA COM DROPOUT
        x = Dense(units = 64, activation = 'sigmoid')(states)
        x = Dropout(rate = 0.1)(x)
        
        # CRIAÇÃO DA SEGUNDA CAMADA OCULTA COM DROPOUT
        y = Dense(units = 32, activation = 'sigmoid')(x)
        y = Dropout(rate = 0.1)(y)
        
        # CRIAÇÃO DA CAMADA DE SAÍDA, CONECTADA COM A ÚLTIMA CAMADA OCULTA
        q_values = Dense(units = number_actions, activation = 'softmax')(y)
        
        # AGREGAR TODAS AS CAMADAS EM UM MODELO (OBJETO MODEL)
        self.model = Model(inputs = states, outputs = q_values)
        
        # COMPILAÇÃO DO MODELO, UTILIZANDO FUNÇÃO DE ERRO E OTIMIZADOR
        self.model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))
