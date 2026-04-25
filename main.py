import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import traceback  # 引入追踪模块，抓取致命错误

app = FastAPI(title="TabICL AI Engine")

# 跨域配置
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# 强行锁定当前文件所在的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Final_Clinical_Practicality_Arena")

# 预先定义变量
imputer = None
scaler = None
model = None
features = None

# 加载模型文件
try:
    print(f"⏳ 正在尝试从绝对路径加载模型: {MODEL_DIR}")
    imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "Scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "Champion_Model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl"))
    print("✅ 武器库挂载成功！模型加载完毕！")
except Exception as e:
    print(f"❌ 致命错误：加载失败: {e}")
    print(f"🔍 当前根目录 {BASE_DIR} 下的文件有：", os.listdir(BASE_DIR))
    if os.path.exists(MODEL_DIR):
        print(f"📂 模型文件夹 {MODEL_DIR} 内部的文件有：", os.listdir(MODEL_DIR))

# 托管网页静态文件
@app.get("/")
async def read_index():
    index_path = os.path.join(BASE_DIR, 'index.html')
    return FileResponse(index_path)

class PatientData(BaseModel):
    ALT: float
    TG: float
    BMI: float
    UA: float
    Glucose: float

@app.post("/predict_nafld")
async def predict_nafld(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="模型未能成功加载，请检查后台日志。")
    try:
        # Pydantic V2 规范写法
        input_df = pd.DataFrame([patient.model_dump()], columns=features)
        
        # 预处理与预测
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        probability = model.predict_proba(X_std)[0][1]
        
        return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
    except Exception as e:
        # 🔥 核心抓虫逻辑：打印详细的崩溃栈追踪
        error_msg = traceback.format_exc()
        print(f"\n{'='*40}")
        print(f"⚠️ 预警！大模型推理时发生崩溃！")
        print(f"详细错误日志如下:\n{error_msg}")
        print(f"{'='*40}\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Render 云端推荐配置
    uvicorn.run(app, host="0.0.0.0", port=10000)
