import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
plt.rcParams['font.family'] = 'Liberation Sans'

import seaborn as sns
import kagglehub
import os

#!pip install pycaret
import pycaret
from pycaret.classification import *

# descargar la última versión del archivo
path = kagglehub.dataset_download("kartik2112/fraud-detection")

st.title("Análisis Exploratorio de Datos")
csv_file_path = os.path.join(path, 'fraudTrain.csv')
# cargar el dataset en un DataFrame
df = pd.read_csv(csv_file_path, index_col=0)

st.divider()

st.subheader("Tabla de datos (primeras 10 filas)")
st.dataframe(df.head(10))

st.divider()

df = df.dropna()
df = df.drop_duplicates()
df = df.drop(['trans_num', 'unix_time'], axis=1)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])

st.subheader("Distribución de transacciones fraudulentas y transacciones no fraudulentas")
fraudulent_count = df[df['is_fraud'] == 1].shape[0]
non_fraudulent_count = df[df['is_fraud'] == 0].shape[0]

# Crear un gráfico de barras
fig, ax = plt.subplots()
ax.bar(['Fraudulentas', 'No Fraudulentas'], [fraudulent_count, non_fraudulent_count])
ax.set_ylabel('Cantidad')
ax.set_title('Distribución de Transacciones Fraudulentas y No Fraudulentas')

st.pyplot(fig)

# Mostrar los valores en texto también
st.write(f"Transacciones fraudulentas: {fraudulent_count}")
st.write(f"Transacciones no fraudulentas: {non_fraudulent_count}")

st.divider()
st.subheader("Distribución de clases (en porcentaje)")
distribucion = df['is_fraud'].value_counts(normalize=True) * 100
st.write(distribucion)