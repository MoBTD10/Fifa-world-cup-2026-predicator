from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import joblib
import numpy as np
app = FastAPI() 
encoder = joblib.load('encoder.pkl')
model = joblib.load('model.pkl')
team_rank = joblib.load('team_rank_dict.pkl')

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




@app.post('/')
def predict(data : PredictionReq):
    home_team = data.home_team
    away_team = data.away_team

    home_rank = team_rank.get(home_team ,50)
    away_rank = team_rank.get(away_team ,50)

    rank_diff = home_rank - away_rank
    is_net = 1
    featuers = np.array([[rank_diff , is_net]],dtype=float)

    pred_encoded = model.predict(featuers)[0]
    team = encoder.inverse_transform([pred_encoded])[0]
    if team == "home":
        winner = home_team
        status_message = "winner"
    elif team == "away":
        winner = away_team
        status_message = "winner"
    else:
        winner = "تعادل (Draw)"
        status_message = "draw"
        
    return {
        "status": "success",
        "match": f"{home_team} vs {away_team}",
        "predicted_winner": winner,
        "result_type": status_message
    }
    