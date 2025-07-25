import json
import os

import joblib
import lightgbm as lgb
import pandas as pd

print(os.path.exists("../best_model.txt"))
print(os.getcwd())
# Load model components
model = lgb.Booster(model_file="../best_model.txt")
# model = joblib.load('model/best_model.pkl')

scaler = joblib.load("model/scaler.pkl")
with open("../selected_features.json") as f:
    selected_features = json.load(f)["selected_features"]
print(selected_features)
