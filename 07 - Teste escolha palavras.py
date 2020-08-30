# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:10:33 2020

@author: Flávia
"""
#%% Change dir path
import os
workdir_path = 'C:/Users/Flávia/Google Drive/' + '00 PUC BI MASTER/00 - PROJ (TCC)/PREDIÇÃO ATIVIDADES/event2mind'  # Inserir o local da pasta onde estão os arquivos de entrada (treino e teste)
os.chdir(workdir_path)
#%%Parametros de Input teste
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
#from sklearn import metrics
from tensorflow.keras.models import load_model
#%%Load modelos escolhidos
#Modelos keras
clf_lstm_model_intencoes = load_model('modelo3C.h5')
clf_lstm_model_emocoes = load_model('modelo4C.h5')
clf_lstm_model_polaridade = load_model('modelo5C.h5')
doc2vec_model = Doc2Vec.load("Event2Mind_Events.model")

#%% arrays intenções e emoções
listaIntencoes = ['get', 'make', 'help', 'see', 'show', 'know', 'go', 'give', 'take', 'keep', 'fell', 'enjoy', 'work', 'wants', 'find', 'look', 'avoid', 'others']
listaEmocoes = ['happy', 'glad', 'satisfied', 'excited', 'relieved','proud','great', 'bad' ,'sad', 'better', 'tired', 'others']

#%%Parametros de Input teste
eventos_exemplos = ["asks for help"]
embeddings_novo = doc2vec_model.infer_vector(eventos_exemplos)
#%% transposed
emb_transposed = embeddings_novo.reshape((1, 400))
#%%normalizando
minEmb = np.min(emb_transposed)
maxEmb = np.max(emb_transposed)
rangeEmb = maxEmb - minEmb
novoMin = -1
novoMax = 1
novoRange = (novoMax - novoMin)
emb_transposed_norm = (((emb_transposed - minEmb) / rangeEmb) * novoRange) + novoMin

#%%Parametros de Input teste
X = np.array(emb_transposed_norm)
X_dev = np.reshape(X, (X.shape[0], 1, X.shape[1]))

#%% teste predict classes
y_predicted_intencoes_index_classe = clf_lstm_model_intencoes.predict_classes(X_dev, batch_size=150)
y_predicted_emocoes_index_classe  = clf_lstm_model_emocoes.predict_classes(X_dev, batch_size=150)

#%%criando array
y_predicted_intencoes = listaIntencoes[y_predicted_intencoes_index_classe[0]]
y_predicted_emocoes = listaEmocoes[y_predicted_emocoes_index_classe[0]]
y_predicted_polaridade = clf_lstm_model_polaridade.predict_classes(X_dev, batch_size=150)[0]
#%% print
print(y_predicted_intencoes)
print(y_predicted_emocoes)
print(y_predicted_polaridade)