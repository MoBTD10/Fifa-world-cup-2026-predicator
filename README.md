# FIFA World Cup 2026 Match Predictor 🏆⚽

A Machine Learning pipeline and FastAPI backend service to forecast match outcomes for the FIFA World Cup 2026. The project trains an **XGBoost Classifier** on historical international match records and FIFA world rankings.

---

## 📁 Repository Structure

```directory
├── data_preprocess.ipynb    # Data cleaning & feature engineering (rolling averages, form, rank diff)
├── model.ipynb              # Model training, hyperparameter tuning & evaluation (XGBoost)
├── data.csv                 # Processed dataset with engineered features
├── Xgb_model.pkl            # Serialized XGBoost model
│
└── site/                    # Web service deployment directory
    ├── app.py               # FastAPI backend app
    ├── requirements.txt     # Python backend dependencies
    └── Procfile             # Render/Heroku deployment config
```

---

## 📊 Machine Learning Pipeline

### 1. Data Preprocessing & Features (`data_preprocess.ipynb`)
- Combines international match records (post-2023) and FIFA World Rankings.
- Computes **5-match rolling performance averages** for both teams:
  - `home_rolling_scored_5` / `away_rolling_scored_5` (Average goals scored)
  - `home_rolling_conceded_5` / `away_rolling_conceded_5` (Average goals conceded)
- Inputs: Team FIFA rankings (`home_team_rank`, `away_team_rank`) and match environment (`neutral` venue).

### 2. Model Training & Evaluation (`model.ipynb`)
- **Target Classes**: `0` (Away Win), `1` (Draw), `2` (Home Win).
- **Model**: An `XGBClassifier` tuned with `GridSearchCV` achieving **61% accuracy** (baseline Logistic Regression also scored 61%).
- **Export**: Saved using `joblib` as `Xgb_model.pkl` for production use.

---

## 🚀 FastAPI Backend (`/site`)

The API dynamically loads the trained model, queries the latest statistics from `data.csv` for the requested teams, and returns prediction details.

### Endpoint: `POST /predict`
- **Request Payload**:
  ```json
  {
    "home_team": "Argentina",
    "away_team": "France",
    "neutral": true
  }
  ```
- **Response Payload**:
  ```json
  {
    "status": "success",
    "match": "Argentina vs France",
    "predicted_winner": "Argentina"
  }
  ```

### Local Setup
1. Navigate to the `site/` folder:
   ```bash
   cd site
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI development server:
   ```bash
   uvicorn app:app --reload
   ```
4. Access the interactive API docs at `http://127.0.0.1:8000/docs`.
