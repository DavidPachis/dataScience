import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

data_train = pd.read_json(
    'https://raw.githubusercontent.com/DavidPachis/dataScience/main/taller4/data/DataSet_Entrenamiento_v2.json')
data_pre = pd.read_json('https://raw.githubusercontent.com/DavidPachis/dataScience/main/taller4/data'
                        '/DataSet_Prediccion.json')

st.title('Taller 4, churn Rate')


def load_data():
    uploaded_file = st.file_uploader(label='upload dataset for training')
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        st.write(data)


if st.checkbox('check for use first model'):
    # load model
    url = 'https://github.com/DavidPachis/dataScience/raw/main/taller4/model/my_model.pkl'
    response = requests.get(url)
    open("my_model.pkl", "wb").write(response.content)
    best_model = joblib.load("my_model.pkl")

if st.button('Make Prediction'):
    inputs = data_pre
    prediction = best_model.predict(inputs)
    print("final prediction", np.squeeze(prediction, -1))
    final_d = np.array2string(prediction)
    st.write(f"Your fares: {final_d}g")

load_data()
