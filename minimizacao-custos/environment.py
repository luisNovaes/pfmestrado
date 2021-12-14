# Estudo de Caso 2 - Minimizando Custos de Consumo de Energia de um Data Center

# Criação do ambiente

# Importação das bibliotecas
import numpy as np

# CRIAÇÃO DO AMBIENTE EM UMA CLASSE
class Environment(object):
    
    # DEFINIÇÃO E INICIALIZAÇÃO DE TODOS OS PARÂMETROS E VARIÁVEIS DO AMBIENTE
    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month = 0,
                 initial_number_users = 10, initial_rate_data = 60):
        self.monthly_atmospheric_temperature = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.initial_month = initial_month
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[initial_month]
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
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
        
    # CRIAÇÃO DO MÉTODO PARA ATUALIZAR O AMBIENTE DEPOIS QUE UMA NOVA AÇÃO É EFETUADA
    def update_env(self, direction, energy_ai, month):
        
        # OBTENDO A RECOMPENSA
                
        # Cálculo da energia gasta quando não tem IA
        energy_noai = 0
        if (self.temperature_noai < self.optimal_temperature[0]):
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
        elif (self.temperature_noai > self.optimal_temperature[1]):
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]
        
        # Cálculo da recompensa
        self.reward = energy_noai - energy_ai
        
        # Escalonamento da recompensa - 0.001
        self.reward = 1e-3 * self.reward
        
        # OBTENDO O PRÓXIMO ESTADO
        
        # Atualização da temperatura atmosférica
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[month]
        
        # Atualização do número de usuários
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        if (self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        elif (self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users
        
        # Atualização da taxa de transmissão
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        if (self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_date
        elif (self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
        
        # Cálculo do delta da temperatura intrínsica
        past_intrinsic_temperature = self.intrinsic_temperature
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature
        
        # Cálculo do delta da temperatura regulada por IA
        if (direction == -1):
            delta_temperature_ai = -energy_ai
        elif (direction == 1):
            delta_temperature_ai = energy_ai
        
        # Atualização da nova temperatura do servidor quando tem IA
        self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai
        
        # Atualização da nova temperatura do servidor quando não tem IA
        self.temperature_noai += delta_intrinsic_temperature
        
        # VERIFICAÇÃO DO FINAL (GAME_OVER)
        if (self.temperature_ai < self.min_temperature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temperature[0]
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
        elif (self.temperature_ai > self.max_temperature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temperature[1]
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
                
        # ATUALIZAÇÃO DOS OBJETIVOS
        
        # Atualização do total de energia gasta pela IA
        self.total_energy_ai += energy_ai
        
        # Atualização do total de energia gasta quando não tem IA
        self.total_energy_noai += energy_noai
        
        # ESCALONAMENTO DOS VALORES DO PRÓXIMO ESTADO
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        next_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        
        # RETORNO DO PRÓXIMO ESTADO, DA RECOMPENSA E DO GAME_OVER
        return next_state, self.reward, self.game_over
        
    # FUNÇÃO PARA RESETAR (REINICIAR O AMBIENTE)    
    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[new_month]
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
        
    # FUNÇÃO PARA RETORNAR O ESTADO ATUAL, A ÚLTIMA RECOMPENSA E SE É GAME_OVER
    def observe(self):
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        return current_state, self.reward, self.game_over
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        