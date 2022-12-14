import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

st.header("Taxi Fare Prediction")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("https://raw.githubusercontent.com/DavidPachis/dataScience/main/proyecto/data/nueva_entrada.csv")

# load model
URL = 'https://github.com/DavidPachis/dataScience/raw/main/proyecto/models/model_Cap.pkl'
response = requests.get(URL)
open("model_Cap.pkl", "wb").write(response.content)
best_xgboost_model = joblib.load('model_Cap.pkl')

if st.checkbox('Show Training Dataframe'):
    data

if st.button('Make Prediction'):
    inputs = data
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    final_d=np.array2string(prediction)
    st.write(f"Your fares: {final_d}g")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
