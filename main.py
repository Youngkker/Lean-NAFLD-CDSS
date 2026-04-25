import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="TabICL 本地纯净版引擎")

# 开启跨域，确保网页能连接到后台
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 🔥 这里写入了您确认过的绝对路径，确保精准加载
MODEL_DIR = r"E:\文献及课题设计\数据库\脂肪肝机器学习写作\Final_Clinical_Practicality_Arena"

# 直接加载模型文件
try:
    print(f"⏳ 正在从路径加载模型: {MODEL_DIR}")
    imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "Scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "Champion_Model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl"))
    print("✅ 模型加载完成！")
except Exception as e:
    print(f"❌ 加载失败，请检查路径下是否有这4个文件: {e}")


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
        # 将传入数据转换为 DataFrame
        input_df = pd.DataFrame([patient.dict()], columns=features)

        # 预处理流程
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)

        # 核心推理
        probability = model.predict_proba(X_std)[0][1]

        return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
    except Exception as e:
        print(f"推理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 本地 AI 后端引擎已点火，正在监听 127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
