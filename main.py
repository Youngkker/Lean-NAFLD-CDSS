import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="TabICL AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "Final_Clinical_Practicality_Arena"
imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "Scaler.pkl"))
champion_model = joblib.load(os.path.join(MODEL_DIR, "Champion_Model.pkl"))
features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl")) 

class PatientData(BaseModel):
    ALT: float
    TG: float
    BMI: float
    UA: float
    Glucose: float

@app.post("/predict_nafld")
async def predict_nafld(patient: PatientData):
    try:
        input_df = pd.DataFrame([patient.dict()], columns=features)
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        probability = champion_model.predict_proba(X_std)[0][1]
        return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🔥 把前端网页丢给访问者的核心代码
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
