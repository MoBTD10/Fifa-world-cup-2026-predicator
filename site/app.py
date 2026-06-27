from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import joblib
import numpy as np
import pandas as pd


app = FastAPI() 
model = joblib.load('Xgb_model.pkl')
data_ = pd.read_csv('data.csv')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class PredictionReq(BaseModel):
    home_team : str
    away_team : str
    neutral: bool


@app.post('/predict')
def predict(data : PredictionReq):
        home_lines = data_[data_['home_team'] == data.home_team]
        away_lines = data_[data_['away_team'] == data.away_team]
        X_input = [
              float(home_lines['home_rolling_scored_5'].iloc[-1]) if not home_lines.empty else 1.0,
              float(home_lines['home_rolling_conceded_5'].iloc[-1]) if not home_lines.empty else 1.0,
              float(away_lines['away_rolling_scored_5'].iloc[-1]) if not away_lines.empty else 1.0,
              float(away_lines['away_rolling_conceded_5'].iloc[-1]) if not away_lines.empty else 1.0,
              float(home_lines['home_rank'].iloc[-1]) if not home_lines.empty else 1.0,
              float(away_lines['away_rank'].iloc[-1]) if not away_lines.empty else 1.0 , 
              1.0 if data.neutral else 0.0
        ]
        pred_num = model.predict([X_input])[0]
        if pred_num == 2:
              winner = data.home_team
        elif pred_num == 0:
              winner = data.away_team
        else:
              winner='Draw'
        return {'winner' : winner}