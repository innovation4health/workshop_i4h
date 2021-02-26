import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
import PIL
from bokeh.models.widgets import Div
from sklearn.preprocessing import StandardScaler
import category_encoders as ce 
import keras
import json
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from functions.utils import *
from tensorflow.keras import backend as K 
import base64

st.set_option('deprecation.showfileUploaderEncoding', False)

def main():

    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    activities = ["Home", "Risco Cardíaco", "Sobre I4H"]
    choice = st.sidebar.selectbox("Menu", activities)

    # ============================== HOME ======================================================= #
    if choice == "Home":

        st.header("Olá a todos! Olá Nelio e Luis")
        st.subheader("Muito obrigado pela sua participação neste evento.")
        st.write("Este aplicativo irá calcular o % de risco do paciente ter uma Fibrilação Atrial (Arritmia Cardíaca) com base em seu histórico de sinais ECG. ")
        st.write("\n\n\n")
        
        image = PIL.Image.open("images/I4H.png")
        st.image(image,caption="")

    # ============================== RISCO CARDIACO ======================================================= #
    if choice == "Risco Cardíaco":
        sub_activities = ["Previsão"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)
        
        if sub_choice == "Previsão":    

            # Extrai o conteúdo do arquivo
            uploaded_file = False

            if st.checkbox('Você quer fazer o upload dos dados para previsão?'):
                uploaded_file = st.file_uploader("Escolha um arquivo .MAT", type="mat")

            if st.button('Processar'):

                if uploaded_file:
                    sinais = loadmat(uploaded_file)

                    sinais_mat = sinais['val']

                    # Extraindo o Batimento Cardiaco do Paciente
                    for channelid, channel in enumerate(sinais_mat):
                        resultado = ecg.ecg(signal = channel, sampling_rate = 300, show = False)
                        heart_rate = np.zeros_like(channel, dtype = 'float')
                        heart_rate = resultado['heart_rate']    
                        
                        st.write('BPM máximo: ', max(heart_rate).round(0))
                        
                        try:
                            if max(heart_rate) > 130:
                                HR = 1
                            else:
                                HR = 0
                        except:
                            continue    

                    # Carregamos o modelo   
                    modelo = load_model('fe_heart_sensor/model/ResNet_30s_34lay_16conv.hdf5')

                    # Valores constantes
                    frequencia = 300
                    tamanho_janela = 30 * frequencia

                    # Fazendo a previsao
                    x = processamento(sinais_mat, tamanho_janela)

                    # Previsões com o modelo (retorna as probabilidades)
                    prob_x, ann_x = previsoes(modelo, x)

                    # Realizando as previsoes
                    x = processamento(sinais_mat, tamanho_janela)
                    prob_x, ann_x = previsoes(modelo, x)
                    st.write('Probabilidade FA (%): ', (prob_x[0, 0] * 100).round(2))

                    # Dataframe para o risco estratificado
                    df_risco = pd.DataFrame({'Probabilidade':[prob_x[0, 0]], 'HR':HR})
                    df_risco['Risco'] = df_risco.apply(classifica_risco, axis = 1)
                    st.write('Risco: ', df_risco['Risco'][0])

                    # Plot
                    x_axis = np.linspace(0., float(len(sinais_mat[0]) / 300), num = len(sinais_mat[0]))
                    plt.rcParams.update({'font.size': 14})
                    fig, ax = plt.subplots(figsize = (16,5))

                    ax.plot(x_axis, sinais_mat[0], 'blue')
                    ax.axis([0, len(sinais_mat[0]) / 300, -2200, 2200])

                    ax.set_title('ECG Paciente')
                    ax.set_xlabel("Tempo (em segundos)")
                    ax.set_ylabel("Milli Volts")

                    st.write(fig)
    
    if choice == 'Sobre I4H':
        st.markdown('<style>body .fullScreenFrame > div { display: flex; justify-content: center; }</style>', unsafe_allow_html=True)
        
        image2 = PIL.Image.open("images/quem_somos.png")
        st.image(image2, caption="", width=700)

if __name__ == '__main__':
    main()