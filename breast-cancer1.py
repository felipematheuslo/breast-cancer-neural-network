"""
Importação dos dados
"""

import pandas as pd

entradas = pd.read_csv('entradas_breast.csv')
saidas = pd.read_csv('saidas_breast.csv')


"""
Separação dos conjuntos de treino e teste (25%)
"""

from sklearn.model_selection import train_test_split

entradas_train, entradas_test, saidas_train, saidas_test = train_test_split(entradas, saidas, test_size=0.25)


"""
Treinamento da rede neural usando o conjunto de treino
"""

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

rede_neural = Sequential()

rede_neural.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
rede_neural.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
rede_neural.add(Dense(units=1, activation='sigmoid'))

otimizador = keras.optimizers.Adam(lr=0.01, decay=0.001)

rede_neural.compile(otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

rede_neural.fit(entradas_train, saidas_train, batch_size=10, epochs=100)

pesos0 = rede_neural.layers[0].get_weights()
pesos1 = rede_neural.layers[1].get_weights()
pesos2 = rede_neural.layers[2].get_weights()

"""
Teste da rede neural usando o conjunto de teste
"""

previsoes = rede_neural.predict(entradas_test)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

acuracia = accuracy_score(saidas_test, previsoes)
matriz = confusion_matrix(saidas_test, previsoes)
resultado = rede_neural.evaluate(entradas_test, saidas_test)
