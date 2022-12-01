import pandas as pd
import numpy as np
import wget
import joblib


data = pd.read_csv("https://raw.githubusercontent.com/DavidPachis/dataScience/main/proyecto/data/nueva_entrada.csv")
# load model
url2 = 'https://github.com/DavidPachis/dataScience/raw/main/proyecto/models/model_Cap.pkl'
path = '/'
model = wget.download(url2, out=path)
own_model = joblib.load("model_Cap.pkl")
best_xgboost_model = own_model

prediction = best_xgboost_model.predict(data)
print("final pred", np.squeeze(prediction,-1))


