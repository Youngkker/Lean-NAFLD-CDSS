import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import traceback

app = FastAPI(title="TabICL AI Engine")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Final_Clinical_Practicality_Arena")

imputer = None
scaler = None
model = None
features = None

# 加载模型
try:
    import tabicl
    imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "Scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "Champion_Model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl"))
except Exception as e:
    print(f"❌ 加载失败: {e}")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

class PatientData(BaseModel):
    ALT: float
    TG: float
    BMI: float
    UA: float
    Glucose: float

@app.post("/predict_nafld")
async def predict_nafld(patient: PatientData):
    try:
        # 1. 准备数据
        input_data = patient.model_dump()
        input_df = pd.DataFrame([input_data])[features]
        
        # 2. 预处理
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        
        # 3. 尝试预测
        probability = model.predict_proba(X_std)[0][1]
        
        return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
        
    except Exception as e:
        # 🔥 乾坤大挪移：拦截崩溃，把真正的错误信息直接显示到网页上！
        error_type = type(e).__name__
        error_detail = str(e)
        return {"prediction": {"risk_probability": f"【后台报错】{error_type}: {error_detail}"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
