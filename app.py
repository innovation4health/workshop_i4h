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

    activities = ["Home", "Risco Cardiaco", "Sobre"]
    choice = st.sidebar.selectbox("Menu", activities)

    # ============================== HOME ======================================================= #
    if choice == "Home":

        st.header("Olá a todos! \nMuito obrigado pela sua participação neste evento.\n")
        st.subheader("Features")
        st.write("- Prevendo Risco Cardíaco com IoT")
        
        image = PIL.Image.open("images/doctor-robot.png")
        #st.image(image,caption="")


    # ============================== RISCO CARDIACO ======================================================= #
    if choice == "Risco Cardiaco":
        sub_activities = ["Predict"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)

        if sub_choice == "Predict":    

            # Extrai o conteúdo do arquivo
            filename = st.text_input('Enter a file path:')
            try:
                with open(filename) as input:
                    st.text(input.read())
            except FileNotFoundError:
                st.error('File not found.')


            sinais = loadmat('fe_heart_sensor/dados/paciente_rodrigo.mat')
            sinais_mat = sinais['val']

            # Extraindo o Batimento Cardiaco do Paciente
            for channelid, channel in enumerate(sinais_mat):
                resultado = ecg.ecg(signal = channel, sampling_rate = 300, show = False)
                heart_rate = np.zeros_like(channel, dtype = 'float')
                heart_rate = resultado['heart_rate']    
                
                st.write('BPM máximo: ', max(heart_rate))  
                
                try:
                    if max(heart_rate) > 130:
                        HR = 1
                    else:
                        HR = 0
                except:
                    continue    

            # Fazendo a previsao
            # Carregamos o modelo   
            modelo = load_model('fe_heart_sensor/model/ResNet_30s_34lay_16conv.hdf5')

            # Valores constantes
            frequencia = 300
            tamanho_janela = 30 * frequencia

            # Fazendo a previsao
            x = processamento(sinais_mat, tamanho_janela)

            # Previsões com o modelo (retorna as probabilidades)
            prob_x, ann_x = previsoes(modelo, x)

            # Lista de classes
            #classes = ['A', 'N', 'O', '~']

            x = processamento(sinais_mat, tamanho_janela)
            prob_x, ann_x = previsoes(modelo, x)
            st.write('Probabilidade FA: ', prob_x[0, 0])

            # Dataframe para o risco estratificado
            df_risco = pd.DataFrame({'Probabilidade':[prob_x[0, 0]], 'HR':HR})
            df_risco['Risco'] = df_risco.apply(classifica_risco, axis = 1)
            st.write('Risco: ', df_risco['Risco'][0])

            # Plot
            x_axis = np.linspace(0., float(len(sinais_mat[0]) / 300), num = len(sinais_mat[0]))
            plt.rcParams.update({'font.size': 14})
            fig, ax = plt.subplots(figsize = (16,5))

            ax.plot(x_axis, sinais_mat[0], 'magenta')
            ax.axis([0, len(sinais_mat[0]) / 300, -2200, 2200])

            ax.set_title('ECG Paciente')
            ax.set_xlabel("Tempo (em segundos)")
            ax.set_ylabel("Milli Volts")

            st.write(fig)
    
    if choice == 'Sobre':
        st.markdown("### Who I am")
        st.write(" - Hello Folks! I am your Doctor Health and my goal is to offer the best experience for you.")
        
        if st.button("website"):
            js = "window.open('https://www.linkedin.com/in/rodrigolima82/')"
            html = '<img src onerror="{}">'.format(js)
            div = Div(text=html)
            st.bokeh_chart(div)      

if __name__ == '__main__':
    main()