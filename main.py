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
    print("✅ 成功找到 tabicl 模块")
    imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "Scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "Champion_Model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl"))
    print("✅ 武器库挂载成功！模型加载完毕！")
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
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        # 1. 准备数据 (完全按照您的本地格式还原)
        input_data = patient.model_dump()
        input_df = pd.DataFrame([input_data])[features]
        
        # 2. 预处理
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        
        # 3. 经典 sklearn 包装器预测模式！
        probability = model.predict_proba(X_std)[0][1]
        
        return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
        
    except Exception as e:
        error_msg = traceback.format_exc()
        # 🔥 只要崩溃，这里一定会打印具体的病历单！
        print(f"\n{'='*40}")
        print(f"⚠️ 推理崩溃详情:\n{error_msg}")
        print(f"{'='*40}\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
