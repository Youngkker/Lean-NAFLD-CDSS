import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="TabICL 云端引擎")

# 开启跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 🔥 核心修正：使用相对路径，确保在 GitHub/Render 上也能找到文件夹
# 确保您的文件夹 'Final_Clinical_Practicality_Arena' 就在 main.py 同级目录
MODEL_DIR = "Final_Clinical_Practicality_Arena"

# 直接加载模型文件
try:
    print(f"⏳ 正在加载模型...")
    imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "Scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "Champion_Model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl"))
    print("✅ 模型加载完成！")
except Exception as e:
    print(f"❌ 加载失败，请检查模型文件是否在 {MODEL_DIR} 文件夹内: {e}")

# 托管网页静态文件
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
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载成功")
    try:
        # 使用 model_dump() 替代 dict() 以兼容 Pydantic V2
        input_df = pd.DataFrame([patient.model_dump()], columns=features)
        
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        
        probability = model.predict_proba(X_std)[0][1]
        
        return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 使用 0.0.0.0 确保云端服务器能正常访问
    uvicorn.run(app, host="0.0.0.0", port=10000)
