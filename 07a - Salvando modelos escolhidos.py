# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 20:16:52 2020

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
cluster_output = pd.read_csv('clusterIntencoes.csv', sep = ',')
cluster_output.rename(columns={"Xintent": "Xintent_2"}, inplace = True)
cluster_output.rename(columns={"cluster_intents": "cluster_output"}, inplace = True)
campo_output = "Xintent_2"
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
listPalavras = ['get', 'make', 'help', 'see', 'show', 'know', 'go', 'give', 'take', 'keep', 'fell', 'enjoy', 'work', 'wants', 'find', 'look', 'avoid']

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
#%% definindo target final (cluster, classes, Xsent ou Osent)
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

#%%criar um embeddings final que segue a ordem do input no dtPrincipal
embeddingDupl = []
cluster = []
descricaoEvento = []

for i in range(1, len(dtPrincipal)):

  linhaEmbedding = (dtPrincipal['input_index'][i]) - 1
  embeddingDupl.append(embeddings_input[linhaEmbedding])
  cluster.append(dtPrincipal[target][i]) #mudar para classes caso queira usar a classe de palavras
  
  
  descricaoEvento.append(dtPrincipal['Event'][i])
  
  
  
#%% criando numpy arrays para os modelos do sci-kit learn 
import numpy as np

X = np.array(embeddingDupl)
Y = np.array(cluster)
X.shape
#%% Oversampling
#pip install imblearn
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_ros, Y_ros = ros.fit_sample(X, Y)

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
x_train, x_test, y_train, y_test = train_test_split(X_ros, Y_ros, random_state=42, stratify=Y_ros, test_size=0.94)

#%% cont_classes
import collections  
count_classes = collections.Counter(y_test)
print(count_classes)
#%% Normalização e Scaler
'''
scaler = StandardScaler()
scaler_model = scaler.fit(x_train)

x_train_scaled = scaler_model.transform(x_train)
x_test_scaled = scaler_model.transform(x_test)



pcaComp = PCA(n_components = 0.99)
pcaModel = pcaComp.fit(x_train_scaled)
x_train_prepared = pcaModel.transform(x_train_scaled)
x_test_prepared = pcaModel.transform(x_test_scaled)

'''

#testando sem normalização e Scaler - versão b com scaler
x_train_prepared = x_train
x_test_prepared = x_test
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
#%% Reshaping para uso no Keras
X_train = np.reshape(x_train_prepared, (x_train_prepared.shape[0], 1, x_train_prepared.shape[1]))
X_test = np.reshape(x_test_prepared, (x_test_prepared.shape[0], 1, x_test_prepared.shape[1]))
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
#regressor.fit(X_train, y_train, epochs = epocas, batch_size = 32)
#regressor.save('modelo3C.h5')
#%% ////////              Iniciando novo modelo 4C ///////////////////////////
#%%Parametros Output
cluster_output = pd.read_csv('clusterEmotions.csv', sep = ',')
cluster_output.rename(columns={"Xemotion": "Xemotion_2"}, inplace = True)
cluster_output.rename(columns={"cluster_emotions": "cluster_output"}, inplace = True)
campo_output = "Xemotion_2"
#%% transforma em lista
listOutput = cluster_output[campo_output].tolist()
#%% Criando classe
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
#%% definindo target final (cluster, classes, Xsent ou Osent)
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
#%%criar um embeddings final que segue a ordem do input no dtPrincipal
embeddingDupl = []
cluster = []

for i in range(1, len(dtPrincipal)):

  linhaEmbedding = (dtPrincipal['input_index'][i]) - 1
  embeddingDupl.append(embeddings_input[linhaEmbedding])
  cluster.append(dtPrincipal[target][i]) #mudar para classes caso queira usar a classe de palavras

len(embeddingDupl)
#%% criando numpy arrays para os modelos do sci-kit learn 
X = np.array(embeddingDupl)
Y = np.array(cluster)
X.shape
#%% Oversampling
ros = RandomOverSampler()
X_ros, Y_ros = ros.fit_sample(X, Y)

#%% Split da base treino e teste
x_train, x_test, y_train, y_test = train_test_split(X_ros, Y_ros, random_state=42, stratify=Y_ros, test_size=0.94)

#%% Normalização e Scaler
#testando sem normalização e Scaler - versão b com scaler
x_train_prepared = x_train
x_test_prepared = x_test
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
#%% Reshaping para uso no Keras
X_train = np.reshape(x_train_prepared, (x_train_prepared.shape[0], 1, x_train_prepared.shape[1]))
X_test = np.reshape(x_test_prepared, (x_test_prepared.shape[0], 1, x_test_prepared.shape[1]))

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
regressor.fit(X_train, y_train, epochs = epocas, batch_size = 32)
regressor.save('modelo4C.h5')
#%% ////////              Iniciando novo modelo 5 ///////////////////////////
#%% definindo target final (cluster_output, classes, Xsent ou Osent)
target = "Xsent"

#%%Cruzando cluster com todos os eventos
#BASE COM TODAS AS INTENÇÕES E EMOÇÕES
dtPrincipal = pd.read_csv('event2MindClean.csv', sep = ',')
dtPrincipal = dtPrincipal[dtPrincipal[campo_output] != "none"]
dtPrincipal = dtPrincipal[dtPrincipal[campo_output] != "'none'"]
 
#como o target é Xsent vamos filtrar quem tem a informação
dtPrincipal = dtPrincipal[dtPrincipal['Xsent'] > 0]

removeCaracteres(campo_output, dtPrincipal)

dtPrincipal = dtPrincipal.join(input_description.set_index(campo_input), on = campo_input, how = 'inner')
dtPrincipal = cluster_output.join(dtPrincipal.set_index(campo_output), on = campo_output, how = 'inner')  

#criando novo index
dtPrincipal['new_index'] = list(range(len(dtPrincipal.index)))
dtPrincipal['new_index'] = dtPrincipal['new_index'] + 1
dtPrincipal.set_index(['new_index'], inplace = True)

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
#%% balanceamento
ros = RandomOverSampler()
X_ros, Y_ros = ros.fit_sample(X, Y)


#%% Split da base treino e teste
x_train, x_test, y_train, y_test = train_test_split(X_ros, Y_ros, random_state=42, stratify=Y_ros, test_size=0.50)
#%% Normalização e Scaler
#testando sem normalização e Scaler
x_train_prepared = x_train
x_test_prepared = x_test
#%% Obtendo as dimensões x e y
dimensao_x = x_train_prepared.shape[1]
camadas_saida  = len(dtPrincipal[target].unique()) +  1
print("dimensao x: %(dx)s e dimensao y: %(dy)s"% {'dx': dimensao_x, 'dy': camadas_saida} )
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
#%% Reshaping para uso no Keras
X_train = np.reshape(x_train_prepared, (x_train_prepared.shape[0], 1, x_train_prepared.shape[1]))
X_test = np.reshape(x_test_prepared, (x_test_prepared.shape[0], 1, x_test_prepared.shape[1]))
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
regressor.fit(X_train, y_train, epochs = epocas, batch_size = 32)
regressor.save('modelo5C.h5')