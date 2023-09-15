import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

ccf = pd.read_csv("creditcard.csv")

columns = "Time V1 V2 V3 V4 V5 V6 V7 V8 V9 V10 V11 V12 V13 V14 V15 V16 V17 V18 V19 V20 V21 V22 V23 V24 V25 V26 V27 V28 Amount".split()
X = ccf.values
Y = ccf.Class
Y = Y.values.reshape(-1, 1)

# Tamanho do conjunto de teste inicial (6%)
test_size = 0.06

# Dividindo o conjunto de dados em treinamento e teste
random.seed(42)  # Defina uma semente para reprodutibilidade
total_samples = len(X)
test_samples = int(total_samples * test_size)
train_samples = total_samples - test_samples

# Índices aleatórios para o conjunto de teste
test_indices = random.sample(range(total_samples), test_samples)

# Criação do conjunto de treinamento e teste
X_train = []
Y_train = []
X_test = []
Y_test = []

for i in range(total_samples):
    if i in test_indices:
        X_test.append(X[i])
        Y_test.append(Y[i])
    else:
        X_train.append(X[i])
        Y_train.append(Y[i])

# Tamanho do conjunto de validação (metade do tamanho do conjunto de teste)
validation_size = len(X_test) // 2

# Dividindo o conjunto de teste em teste e validação
random.seed(42)  # Defina a mesma semente para reprodutibilidade
validation_indices = random.sample(range(len(X_test)), validation_size)

X_dev = []
Y_dev = []

for i in range(len(X_test)):
    if i in validation_indices:
        X_dev.append(X_test[i])
        Y_dev.append(Y_test[i])
    else:
        X_test[i-validation_size] = X_test[i]
        Y_test[i-validation_size] = Y_test[i]

# Atualizando o tamanho do conjunto de teste após a divisão
test_samples -= validation_size

# Convertendo as listas em arrays NumPy
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test[:test_samples])
Y_test = np.array(Y_test[:test_samples])
X_dev = np.array(X_dev)
Y_dev = np.array(Y_dev)




