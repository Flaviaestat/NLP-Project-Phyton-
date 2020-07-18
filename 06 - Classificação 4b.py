# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 22:53:23 2020

@author: Flávia
"""
#%% Change dir path
import os
workdir_path = 'C:/Users/Flávia/Google Drive/' + '00 PUC BI MASTER/00 - PROJ (TCC)/PREDIÇÃO ATIVIDADES/event2mind'  # Inserir o local da pasta onde estão os arquivos de entrada (treino e teste)
os.chdir(workdir_path)

#%% install gensim
#pip install gensim
#%%Parametros de Input
from gensim.models.doc2vec import Doc2Vec
model = Doc2Vec.load("Event2Mind_Events.model")
import pandas as pd
input_description = pd.read_csv('embeddingsEvents.csv', sep = ';')
campo_input = "Event"
#%%Parametros Output
cluster_output = pd.read_csv('clusterEmotions.csv', sep = ',')
cluster_output.rename(columns={"Xemotion": "Xemotion_2"}, inplace = True)
cluster_output.rename(columns={"cluster_emotions": "cluster_output"}, inplace = True)
campo_output = "Xemotion_2"
#%% Tratamentos da base
input_description['input_index'] = list(range(len(input_description.index)))
input_description['input_index'] = input_description['input_index'] + 1
embeddings_input = model.docvecs.vectors_docs

def removeCaracteres(nomeColuna, dataSet):
  dataSet[nomeColuna]= dataSet[nomeColuna].astype(str)
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace("`` ve", "")
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace("`` s", "")
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace("'", "")
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace("&", " ")
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace(",", " ")
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace("  ", " ")
#%% transforma em lista
listOutput = cluster_output[campo_output].tolist()
#%% Criando classe

#listPalavras = ['get', 'make', 'help', 'see', 'show', 'know', 'go', 'give', 'take', 'keep', 'fell', 'enjoy', 'work', 'wants', 'find', 'look', 'avoid']
listPalavras = ['happy', 'glad', 'satisfied', 'excited', 'relieved','proud','great', 'bad' ,'sad', 'better', 'tired']

import math

classes = []

for i in range(0, len(listOutput)):
  texto = listOutput[i]
  classeTemp = []
  for z in range(0, len(listPalavras)):
    if (texto.find(listPalavras[z]) != -1):
        flag = z
        classeTemp.append(flag)
         
  classeFinal = pd.Series(classeTemp).min()
  classes.append(classeFinal)

maximo = pd.Series(classes).max()    
classes = [ maximo + 1 if math.isnan(x) else x for x in classes] 

cluster_output = cluster_output.assign(classes=classes)

#%% definindo target final (cluster_output, classes, Xsent ou Osent)
target = "classes"

#%%Cruzando cluster com todos os eventos
#BASE COM TODAS AS INTENÇÕES E EMOÇÕES
dtPrincipal = pd.read_csv('event2MindClean.csv', sep = ',')
dtPrincipal = dtPrincipal[dtPrincipal[campo_output] != "none"]
dtPrincipal = dtPrincipal[dtPrincipal[campo_output] != "'none'"]
removeCaracteres(campo_output, dtPrincipal)

dtPrincipal = dtPrincipal.join(input_description.set_index(campo_input), on = campo_input, how = 'inner')
dtPrincipal = cluster_output.join(dtPrincipal.set_index(campo_output), on = campo_output, how = 'inner')  

#criando novo index
dtPrincipal['new_index'] = list(range(len(dtPrincipal.index)))
dtPrincipal['new_index'] = dtPrincipal['new_index'] + 1
dtPrincipal.set_index(['new_index'], inplace = True)

len(dtPrincipal)

#%%criar um embeddings final que segue a ordem do input no dtPrincipal
embeddingDupl = []
cluster = []

for i in range(1, len(dtPrincipal)):

  linhaEmbedding = (dtPrincipal['input_index'][i]) - 1
  embeddingDupl.append(embeddings_input[linhaEmbedding])
  cluster.append(dtPrincipal[target][i]) 
len(embeddingDupl)
#%% criando numpy arrays para os modelos do sci-kit learn 
import numpy as np

X = np.array(embeddingDupl)
Y = np.array(cluster)
X.shape

#%%import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#%% Imports sklearn
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#%% Split da base treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
#%% Normalização e Scaler

scaler = StandardScaler()
scaler_model = scaler.fit(x_train)

x_train_scaled = scaler_model.transform(x_train)
x_test_scaled = scaler_model.transform(x_test)
'''
pcaComp = PCA(n_components = 0.99)
pcaModel = pcaComp.fit(x_train_scaled)
x_train_prepared = pcaModel.transform(x_train_scaled)
x_test_prepared = pcaModel.transform(x_test_scaled)

'''

#testando sem normalização e Scaler
x_train_prepared = x_train_scaled
x_test_prepared = x_test_scaled
#%% Obtendo as dimensões x e y
dimensao_x = x_train_prepared.shape[1]

camadas_saida  = len(dtPrincipal[target].unique())
print("dimensao x: %(dx)s e dimensao y: %(dy)s"% {'dx': dimensao_x, 'dy': camadas_saida} )
#%% nomes e epocas do modelo
epocas = 500

#nomes dos modelos
modelo1 = 'MLP 1 Camada'
modelo2 = 'MLP 2 Camadas'
modelo3 = 'KNN 2'
modelo4 = 'LSTM 1 Camada'
modelo5 = 'LSTM 2 Camadas'
modelo6 = 'LSTM 3 Camadas'
#%% calculando número de neuronios de cada camada modelo LSTM
for k in range(3, 10):
  potencia = 2 ** k
  if potencia < dimensao_x:
    primeiro_neuronio_lstm = potencia #número de neurônios da primeira camada e assim por diante

for k in range(3, 10):
  potencia = 2 ** k
  if potencia < primeiro_neuronio_lstm:
    segundo_neuronio_lstm = potencia

for k in range(3, 10):
  potencia = 2 ** k
  if potencia < segundo_neuronio_lstm:
    terceiro_neuronio_lstm = potencia

print("O primeiro neuronio do modelo de LSTM tera %(n1)s camadas. O segundo neuronio terá %(n2)s camadas. O terceiro neuronio terá %(n3)s camadas." % {'n1': primeiro_neuronio_lstm, 'n2': segundo_neuronio_lstm, 'n3': terceiro_neuronio_lstm})
#%% calculando numero de neuronios de cada camada modelo multilayer perceptron
primeira_hl_mlp = round((dimensao_x + camadas_saida) / 2)
segunda_hl_mlp = round(primeira_hl_mlp /2)

print("O primeiro neuronio do modelo de MLP tera %(n1)s camadas. O segundo neuronio terá %(n2)s camadas" % {'n1': primeira_hl_mlp, 'n2': segunda_hl_mlp})
#%% Modelo Multilayer Perceptron 1 camada
clf_mlp1 = MLPClassifier(solver='sgd'
                     , hidden_layer_sizes=(primeira_hl_mlp)
                     , max_iter = (epocas * 2)
                     , random_state=1)

clf_mlp1.out_activation_ = 'softmax'

clf_mlp1.fit(x_train_prepared, y_train)
y_predicted_mlp1 = clf_mlp1.predict(x_test_prepared)

accuracy_modelo1 = metrics.accuracy_score(y_test, y_predicted_mlp1).round(3)
print(accuracy_modelo1)
#%% salva resultados
mlp1_results = pd.DataFrame(list(zip(y_predicted_mlp1, y_test)), columns =['predito', 'real'])
df_confusion = pd.crosstab(mlp1_results.real, mlp1_results.predito)

export_path = workdir_path + '/matrizConfusaoMLP1.csv'
df_confusion.to_csv (export_path, index = True, header=True)
#%% Modelo Multilayer Perceptron 2 camadas
clf_mlp2 = MLPClassifier(solver='sgd'
                     , hidden_layer_sizes=(primeira_hl_mlp, segunda_hl_mlp)
                     , max_iter = (epocas * 2)
                     , random_state=1)

clf_mlp2.out_activation_ = 'softmax'

clf_mlp2.fit(x_train_prepared, y_train)
y_predicted_mlp2 = clf_mlp2.predict(x_test_prepared)

accuracy_modelo2 = metrics.accuracy_score(y_test, y_predicted_mlp2).round(3)
print(accuracy_modelo2)
#%% Modelo KNN
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(x_train_prepared, y_train)
y_predicted_neigh = neigh.predict(x_test_prepared)
accuracy_modelo3 = metrics.accuracy_score(y_test, y_predicted_neigh).round(3)
print(accuracy_modelo3)
#%% Reshaping para uso no Keras
X_train = np.reshape(x_train_prepared, (x_train_prepared.shape[0], 1, x_train_prepared.shape[1]))
X_test = np.reshape(x_test_prepared, (x_test_prepared.shape[0], 1, x_test_prepared.shape[1]))
#%% Modelo LSTM com uma camada
#Initicializar a RNN
regressor = Sequential()
 
# Adicionar a primeira camada LSTM e Dropout 
regressor.add(LSTM(units = primeiro_neuronio_lstm, input_shape=(1, x_train_prepared.shape[1])))
regressor.add(Dropout(0.2))
 
# camada de saída
regressor.add(Dense(units = camadas_saida, activation='softmax')) #para classificação dar como entrada as classes com função de ativação softmax

# Compilar a rede
regressor.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Visualizar a rede
regressor.summary()
#%% Rodando o regressor
regressor.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = epocas, batch_size = 32)
acur_test = regressor.evaluate(X_test, y_test, verbose=0)
accuracy_modelo4 = acur_test[1]
print(accuracy_modelo4)
#%% salvando resultados
y_predicted_lstm = regressor.predict_classes(X_test)
lstm_results = pd.DataFrame(list(zip(y_predicted_lstm, y_test)), columns =['predito', 'real'])
df_confusion = pd.crosstab(lstm_results.real, lstm_results.predito)
export_path = workdir_path + '/matrizConfusaoLSTM1.csv'
df_confusion.to_csv (export_path, index = True, header=True)
#%% Modelo LSTM com duas camadas
# Initicializar a RNN
regressor = Sequential()
 
# Adicionar a primeira camada LSTM e Dropout 
regressor.add(LSTM(units = primeiro_neuronio_lstm, return_sequences = True, input_shape=(1, x_train_prepared.shape[1])))
regressor.add(Dropout(0.2))
 
# Adicionar a terceira camada LSTM e Dropout
regressor.add(LSTM(units = segundo_neuronio_lstm))
regressor.add(Dropout(0.2))
 
# camada de saída
regressor.add(Dense(units = camadas_saida, activation='softmax')) #para classificação dar como entrada as classes com função de ativação softmax

# Compilar a rede
regressor.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Visualizar a rede
regressor.summary()
#%% Rodando o regressor
regressor.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = epocas, batch_size = 32 )
acur_test = regressor.evaluate(X_test, y_test, verbose=0)
accuracy_modelo5 = acur_test[1]
print(accuracy_modelo5)
#%% salvando resultados
y_predicted_lstm = regressor.predict_classes(X_test)
lstm_results = pd.DataFrame(list(zip(y_predicted_lstm, y_test)), columns =['predito', 'real'])
df_confusion = pd.crosstab(lstm_results.real, lstm_results.predito)

export_path = workdir_path + '/matrizConfusaoLSTM2.csv'
df_confusion.to_csv (export_path, index = True, header=True)
#%% Modelo LSTM com três camadas
# Initicializar a RNN
regressor = Sequential()
 
# Adicionar a primeira camada LSTM e Dropout 
regressor.add(LSTM(units = primeiro_neuronio_lstm, return_sequences = True, input_shape=(1, x_train_prepared.shape[1])))
regressor.add(Dropout(0.2))
 
# Adicionar a segunda camada LSTM e Dropout
regressor.add(LSTM(units = segundo_neuronio_lstm, return_sequences = True))
regressor.add(Dropout(0.2))
 
# Adicionar a terceira camada LSTM e Dropout
regressor.add(LSTM(units = terceiro_neuronio_lstm))
regressor.add(Dropout(0.2))
 
# camada de saída
regressor.add(Dense(units = camadas_saida, activation='softmax')) #para classificação dar como entrada as classes com função de ativação softmax

# Compilar a rede
regressor.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Visualizar a rede
regressor.summary()
#%% Rodando o regressor
regressor.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = epocas, batch_size = 32 )
acur_test = regressor.evaluate(X_test, y_test, verbose=0)
accuracy_modelo6 = acur_test[1]
print(accuracy_modelo6)
#%% Salvando resultados
y_predicted_lstm = regressor.predict_classes(X_test)
lstm_results = pd.DataFrame(list(zip(y_predicted_lstm, y_test)), columns =['predito', 'real'])
df_confusion = pd.crosstab(lstm_results.real, lstm_results.predito)
export_path = workdir_path + '/matrizConfusaoLSTM3.csv'
df_confusion.to_csv (export_path, index = True, header=True)
#%% Resultados gerais dos modelos
descricoes_modelos = [modelo1, modelo2, modelo3, modelo4, modelo5, modelo6]
acuracias = [accuracy_modelo1, accuracy_modelo2, accuracy_modelo3, accuracy_modelo4, accuracy_modelo5, accuracy_modelo6]
acuracias_df = pd.DataFrame(list(zip(descricoes_modelos, acuracias)), 
               columns =['modelo', 'acuracia']) 

export_path = workdir_path + '/acuracias.csv'
acuracias_df.to_csv (export_path, index = True, header=True)

