# -*- coding: utf-8 -*-
"""
Exploração de palavras nos Clusters de Intenções

Created on Wed Jul  1 14:01:04 2020

@author: Flávia
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from nltk.tokenize import word_tokenize
import nltk.data
nltk.download('punkt')

import matplotlib.pyplot as plt
import warnings;
warnings.filterwarnings('ignore')

#%% Change dir path
workdir_path = 'C:/Users/Flávia/Google Drive/' + '00 PUC BI MASTER/00 - PROJ (TCC)/PREDIÇÃO ATIVIDADES/event2mind'  # Inserir o local da pasta onde estão os arquivos de entrada (treino e teste)
os.chdir(workdir_path)

#%% Load datasets
datasetCompleto = pd.read_csv('event2MindClean.csv')
clusterIntencoes = pd.read_csv('clusterIntencoes.csv', sep = ',')
#%% função de tratamento
def removeCaracteres(nomeColuna, dataSet):
  dataSet[nomeColuna]= dataSet[nomeColuna].astype(str)
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace("`` ve", "")
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace("`` s", "")
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace("'", "")
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace("&", " ")
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace(",", " ")
  dataSet[nomeColuna] = dataSet[nomeColuna].str.replace("  ", " ")
  
removeCaracteres('Xintent_2', datasetCompleto)
#%% Trata os dados e junta os dataframes
clusterIntencoes['Xintent_2'] = clusterIntencoes['Xintent']
dtSetWithCluster = clusterIntencoes[["cluster_intents", "Xintent_2"]].join(datasetCompleto.set_index('Xintent_2'), on = 'Xintent_2', how = 'left')  
#%% Função para polaridade
def polaridade_positiva(x):
  if x >= 4:
    return 1
  else:
    return 0

def polaridade_negativa(x):
  if x <= 2:
    return 1
  else:
    return 0
#%% criando flags de polaridades Xsent
dtSetWithCluster['pos_polarity_Xsent'] = dtSetWithCluster['Xsent'].apply(polaridade_positiva) #com apply e função de classificação
dtSetWithCluster['neg_polarity_Xsent'] = dtSetWithCluster['Xsent'].apply(polaridade_negativa) #com apply e função de classificação
#%% criando flags de polaridades Osent
dtSetWithCluster['pos_polarity_Osent'] = dtSetWithCluster['Osent'].apply(polaridade_positiva) #com apply e função de classificação
dtSetWithCluster['neg_polarity_Osent'] = dtSetWithCluster['Osent'].apply(polaridade_negativa) #com apply e função de classificação
#%% Agregando
basePosit_Xsent = dtSetWithCluster.groupby("cluster_intents")["pos_polarity_Xsent"].mean().to_frame()
baseNeg_Xsent = dtSetWithCluster.groupby("cluster_intents")["neg_polarity_Xsent"].mean().to_frame()
basePosit_Osent = dtSetWithCluster.groupby("cluster_intents")["pos_polarity_Osent"].mean().to_frame()
baseNeg_Osent = dtSetWithCluster.groupby("cluster_intents")["neg_polarity_Osent"].mean().to_frame()
#%% Exportando polaridades
media_Xsent = basePosit_Xsent.join(baseNeg_Xsent, on = 'cluster_intents', how = 'left')
media_Osent = basePosit_Osent.join(baseNeg_Osent, on = 'cluster_intents', how = 'left')
media = media_Xsent.join(media_Osent, on = 'cluster_intents', how = 'left')
export_path = workdir_path + '/polaridadesMediaClusterIntencoes.csv'
media.to_csv (export_path, index = True, header=True)
#%% Deduplicando base de intencoes
dtSetWithClusterDedup = dtSetWithCluster[['Xintent_2','cluster_intents']]
dtSetWithClusterDedup.drop_duplicates(keep=False, inplace=True)
#%% Função para contagem de palavras por cluster
def count_words_cluster(df, coluna_cluster):
  contagemClusters = pd.DataFrame(columns=['index', 'count', 'cluster'])
  num_cluster = len(df[coluna_cluster].unique())

  for i in range(0, num_cluster):
     
    filtro_cluster = df[df[coluna_cluster] == i] 
    #  ******* alterar a coluna Xintent_2 ******
    newdf = filtro_cluster.Xintent_2.str.split(expand=True).stack().value_counts().reset_index(name="count").query("count > 5") 
    newdf['cluster'] = i
    contagemClusters = contagemClusters.append(newdf)

  return contagemClusters
#%% chamada da função
dtFrameContagens = count_words_cluster(dtSetWithClusterDedup, 'cluster_intents')
#%% exportação
export_path = workdir_path + '/contagemClusterIntencoes.csv'
dtFrameContagens.to_csv (export_path, index = False, header=True)