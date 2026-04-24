import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

app = FastAPI(title="TabICL AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 极其保守的环境导入（防暴毙）
try:
    import pandas as pd
    import joblib
    print("✅ 依赖库 (pandas/joblib) 导入成功！")
except ImportError as e:
    print(f"❌ 致命错误：依赖库未安装！请检查 requirements.txt 是否拼写正确！报错信息: {e}")

# 2. 全图雷达自动寻找模型（防暴毙）
imputer, scaler, champion_model, features = None, None, None, None
try:
    if os.path.exists("Champion_Model.pkl"):
        MODEL_DIR = "."
        print("🎯 雷达扫描：在【根目录】发现了模型武器库！")
    elif os.path.exists("Final_Clinical_Practicality_Arena/Champion_Model.pkl"):
        MODEL_DIR = "Final_Clinical_Practicality_Arena"
        print("🎯 雷达扫描：在【子文件夹】发现了模型武器库！")
    else:
        raise FileNotFoundError("找不到 .pkl 文件，请确认是否成功上传到了 GitHub！")

    imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "Scaler.pkl"))
    champion_model = joblib.load(os.path.join(MODEL_DIR, "Champion_Model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl")) 
    print("✅ 武器库挂载成功，AI 引擎全面就绪！")
except Exception as e:
    print(f"❌ 武器库挂载失败，但服务器依然存活。报错原因: {e}")

class PatientData(BaseModel):
    ALT: float
    TG: float
    BMI: float
    UA: float
    Glucose: float

@app.post("/predict_nafld")
async def predict_nafld(patient: PatientData):
    if champion_model is None:
        raise HTTPException(status_code=500, detail="模型未成功加载，无法进行预测，请查看后台日志！")
    try:
        input_df = pd.DataFrame([patient.dict()], columns=features)
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        probability = champion_model.predict_proba(X_std)[0][1]
        return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def serve_frontend():
    if not os.path.exists("index.html"):
        return {"error": "找不到 index.html 网页文件，请确认是否成功上传！"}
    return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
