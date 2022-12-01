import wget
import joblib


# load model
url2 = 'https://github.com/DavidPachis/dataScience/raw/main/proyecto/models/model_Cap.pkl' \
       ''
path = '.'
model = wget.download(url2, out=path)
own_model = joblib.load("model_Cap.pkl")
best_xgboost_model= own_model
best_xgboost_model.save_model("best_model.json")
