import joblib
import requests

# load model
URL = 'https://github.com/DavidPachis/dataScience/raw/main/proyecto/models/model_Cap.pkl'
response = requests.get(URL)
open("model_Cap.pkl", "wb").write(response.content)
best_xgboost_model = joblib.load('model_Cap.pkl')
own_model = joblib.load('model_Cap.pkl')
best_xgboost_model= own_model
best_xgboost_model.save_model("best_model.json")
