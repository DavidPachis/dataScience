import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
import wget
import joblib

st.header("Taller 4, churn Rate")
data_train = pd.read_json(
    'https://raw.githubusercontent.com/DavidPachis/dataScience/main/taller4/data/DataSet_Entrenamiento_v2.json')
data_pre = 'https://raw.githubusercontent.com/DavidPachis/dataScience/main/taller4/data/DataSet_Prediccion.json'


def load_data():
    uploaded_file = st.file_uploader(label='upload dataset for training')
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        st.image(data)
    else:
        data = pd.read_json(
            'https://raw.githubusercontent.com/DavidPachis/dataScience/main/taller4/data/DataSet_Prediccion.json')


if st.checkbox('check for use first model'):
    # load model
    url2 = 'https://github.com/DavidPachis/dataScience/raw/main/taller4/model/my_model.pkl'
    path = '/content/sample_data'
    model = wget.download(url2, out=path)
    best_model = joblib.load("/content/model_Cap.pkl")

if st.button('Make Prediction'):
    inputs = data_pre
    prediction = best_model.predict(inputs)
    print("final", np.squeeze(prediction, -1))

