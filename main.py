import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="TabICL AI Engine")

# 跨域配置
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 模型文件夹
MODEL_DIR = "Final_Clinical_Practicality_Arena"

# 加载模型
try:
    imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "Scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "Champion_Model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl"))
except Exception as e:
    print(f"❌ 加载模型失败: {e}")

# 托管网页：访问根目录返回 index.html
@app.get("/")
async def read_index():
    return FileResponse('index.html')

class PatientData(BaseModel):
    ALT: float
    TG: float
    BMI: float
    UA: float
    Glucose: float

@app.post("/predict_nafld")
async def predict_nafld(patient: PatientData):
    try:
        input_df = pd.DataFrame([patient.model_dump()], columns=features)
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        probability = model.predict_proba(X_std)[0][1]
        return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Render 部署建议端口 10000
    uvicorn.run(app, host="0.0.0.0", port=10000)
