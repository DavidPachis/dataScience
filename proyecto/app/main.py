import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
import wget
import joblib

st.header("Fish Weight Prediction App")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("https://raw.githubusercontent.com/DavidPachis/dataScience/main/proyecto/data/nueva_entrada.csv")

# load model
url2 = 'https://github.com/DavidPachis/dataScience/raw/main/proyecto/models/model_Cap.pkl'
path = '/content/sample_data'
model = wget.download(url2, out=path)
best_xgboost_model = joblib.load("/content/model_Cap.pkl")

if st.checkbox('Show Training Dataframe'):
    data

if st.button('Make Prediction'):
    inputs = data
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your fare: {np.squeeze(prediction, -1):.2f}g")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")