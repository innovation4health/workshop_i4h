# Import de bibliotecas
import pandas as pd
import numpy as np
import glob
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Deep Learning
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import load_model
from keras.utils.np_utils import to_categorical

# Imports
import biosppy
from biosppy.signals import ecg

## Função para preparar um dataframe com as feaures de Estratificacao de Riscos
def processamento(x, maxlen):    
    x =  np.nan_to_num(x)
    x =  x[0, 0:maxlen]
    x = x - np.mean(x)
    x = x / np.std(x)
    tmp = np.zeros((1, maxlen))
    tmp[0, :len(x)] = x.T 
    x = tmp
    x = np.expand_dims(x, axis = 2) 
    del tmp
    return x

# Funcao para realizar as previsoes
def previsoes(model, x):    
    prob = model.predict(x)
    ann = np.argmax(prob)
    return prob, ann    

# Função para estratificar o risco
def classifica_risco(registro):

    # Condição 1
    if registro[0] >= 0.8 and registro[1] == 1.0 : 
        return 'Alto'
        
    # Condição 2 
    elif (registro[0] >= 0.6 or registro[0] < 0.8) and registro[1] == 1.0 : 
        return 'Médio'
            
    # Condição 3
    else:
        return 'Baixo'    