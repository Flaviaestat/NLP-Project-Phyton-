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
doc2vec_model = Doc2Vec.load("Event2Mind_Events.model")

#eventos_exemplos = ['give money poor','asks help', 'drives mom', 'buys big house', 'goes out friends', 'makes food children']
eventos_exemplos = ['give money poor']
embeddings_novo = doc2vec_model.infer_vector(eventos_exemplos)


#embeddings_novo = []

'''for i in range(0,len(eventos_exemplos)):
    evento = eventos_exemplos[i]
    embeddings_calc = doc2vec_model.infer_vector(evento)
    embeddings_novo.append(embeddings_calc)
'''    
#%%Transpor
import numpy as np
emb_transposed = embeddings_novo.reshape((1, 400))


#%%Parametros de Input teste

from sklearn.preprocessing import StandardScaler
X = np.array(emb_transposed)
scaler = StandardScaler()
#scaler_model = scaler.fit(X)
scaler_model = load.
x_scaled = scaler_model.transform(X)
x_prepared = x_scaled
X_dev = np.reshape(x_prepared, (x_prepared.shape[0], 1, x_prepared.shape[1]))

#%%import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#%% Imports sklearn
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

#%%Load modelos escolhido
from joblib import dump, load
clf_model_intencoes = load('MLPmodel3.joblib')
#clf_model_emocoes = load('filename.joblib')

#keras
#from keras.models import load_model
#clf_lstm_model_intencoes = load_model('')
#clf_lstm_model_emocoes = load_model('')


#%%predict
y_predicted_intencoes = clf_model_intencoes.predict_classes(X_dev)
y_predicted_emocoes = clf_model_emocoes.predict_classes(X_dev)