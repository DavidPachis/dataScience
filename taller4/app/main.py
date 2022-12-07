import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

data_pre = pd.read_json('https://raw.githubusercontent.com/DavidPachis/dataScience/main/taller4/data'
                        '/DataSet_Prediccion.json')

st.title('Taller 4, churn Rate')


def load_data():
    uploaded_file = st.file_uploader(label='upload dataset for training')
    data_train = pd.read_json(
        'https://raw.githubusercontent.com/DavidPachis/dataScience/main/taller4/data/DataSet_Entrenamiento_v2.json')
    if uploaded_file is not None:
        data_train = uploaded_file.getvalue()
    return data_train


def cleaning(dataset):
    # se limpia la columna TotalChares que tiene problemas con valores 0
    dataset.loc[dataset["tenure"] == 0, "TotalCharges"] = "0"
    dataset['TotalCharges'].astype(float)
    # se asocian por tipo las columnas
    objetivo = dataset["Churn"]
    numericas = dataset[["MonthlyCharges", "tenure", "SeniorCitizen", "TotalCharges"]]
    gender = dataset["gender"]
    categoricas_1 = dataset[['Partner', 'PhoneService', 'PaperlessBilling', 'Dependents']]
    categoricas_2 = dataset[[
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]]
    excluidas = dataset[["customerID", "MultipleLines"]]

    # se hace la transformación
    objetivo = objetivo.replace(['No', 'Yes'], [0, 1])
    gender = gender.replace(['Female', 'Male'], [0, 1])
    categoricas_1 = categoricas_1.replace(['No', 'Yes'], [0, 1])
    categoricas_2 = pd.get_dummies(categoricas_2)
    numericas = numericas.astype(float)

    # se construye todo el dataset limpio de nuevo
    clean_dataset = pd.DataFrame().join([objetivo, gender, categoricas_1, categoricas_2, numericas], how="outer")
    return clean_dataset


def cleaning_1(dataset):
    # se limpia la columna TotalChares que tiene problemas con valores 0
    dataset.loc[dataset["tenure"] == 0, "TotalCharges"] = "0"
    dataset['TotalCharges'].astype(float)
    # se asocian por tipo las columnas
    numericas = dataset[["MonthlyCharges", "tenure", "SeniorCitizen", "TotalCharges"]]
    gender = dataset["gender"]
    categoricas_1 = dataset[['Partner', 'PhoneService', 'PaperlessBilling', 'Dependents']]
    categoricas_2 = dataset[[
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]]
    excluidas = dataset[["customerID", "MultipleLines"]]

    # se hace la transformación
    gender = gender.replace(['Female', 'Male'], [0, 1])
    categoricas_1 = categoricas_1.replace(['No', 'Yes'], [0, 1])
    categoricas_2 = pd.get_dummies(categoricas_2)
    numericas = numericas.astype(float)

    # se construye todo el dataset limpio de nuevo
    clean_dataset = pd.DataFrame().join([gender, categoricas_1, categoricas_2, numericas], how="outer")
    return clean_dataset


if st.checkbox('check for use first model'):
    # load model
    url = 'https://github.com/DavidPachis/dataScience/raw/main/taller4/model/my_model.pkl'
    response = requests.get(url)
    open("my_model.pkl", "wb").write(response.content)
    best_model = joblib.load("my_model.pkl")

if st.button('Make Prediction'):
    inputs = cleaning_1(data_pre)

    prediction = best_model.predict(inputs)
    print("final prediction", np.squeeze(prediction, -1))
    final_d = np.array2string(prediction)
    st.write(f"Your churn: {final_d}g")
load_data()

if st.button('Make Prediction with new model'):
    inputs = data_pre
    prediction = best_model.predict(inputs)
    print("final prediction", np.squeeze(prediction, -1))
    final_d = np.array2string(prediction)
    st.write(f"Your fares: {final_d}g")
