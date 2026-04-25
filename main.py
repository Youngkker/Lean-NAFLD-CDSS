import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import traceback

# 尝试导入 torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

app = FastAPI(title="Lean NAFLD AI Engine")

# 跨域配置
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
STARTUP_ERROR = None

# 加载深度学习模型
try:
    import tabicl
    imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "Scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "Champion_Model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl"))
    
    if HAS_TORCH and hasattr(model, 'eval'):
        model.eval()
except Exception as e:
    STARTUP_ERROR = f"{type(e).__name__}: {str(e)}"
    print(f"❌ 加载失败: {STARTUP_ERROR}")

# 首页加载 & 健康检查
@app.get("/")
@app.head("/")  # 🔥 加上这一行，解决 Render 部署红字报错的核心
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
    if imputer is None or model is None:
        return {"prediction": {"risk_probability": f"【模型加载失败】: {STARTUP_ERROR}"}}

    try:
        # 1. 准备数据
        input_data = patient.model_dump()
        input_df = pd.DataFrame([input_data])
        
        # 2. 自动匹配特征列
        if features is not None:
            # 过滤掉 input_data 中不属于 features 的字段，并按 features 排序
            available_cols = [f for f in features if f in input_df.columns]
            input_df = input_df[available_cols]
            
        # 3. 预处理
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        
        # 4. TabICL 专属推理引擎
        if HAS_TORCH:
            tensor_input = torch.tensor(X_std, dtype=torch.float32).unsqueeze(0)
            
            # 识别 TabICL 深度学习模型
            if type(model).__name__ == 'TabICL' or hasattr(model, 'forward_with_cache'):
                with torch.no_grad():
                    if getattr(model, "has_cache", False):
                        output = model.forward_with_cache(X_test=tensor_input, use_cache=True, return_logits=False)
                    else:
                        y_train_dummy = torch.empty((1, 0), dtype=torch.long)
                        output = model(X=tensor_input, y_train=y_train_dummy, return_logits=False)
                
                probability = output[0, 0, 1].item()
                return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
            
            # 兼容其他普通 Torch 模型
            try:
                with torch.no_grad():
                    output = model(tensor_input)
                if hasattr(torch.nn.functional, 'softmax'):
                    prob_tensor = torch.nn.functional.softmax(output, dim=-1)
                    probability = prob_tensor.flatten()[1].item()
                    return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
            except Exception:
                pass
                
        # 兼容传统机器学习模型
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_std)[0][1]
            return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
            
        raise ValueError("无法解析此模型的输出结构！")
        
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        error_location = f"第 {tb[-1].lineno} 行" if tb else "未知位置"
        return {"prediction": {"risk_probability": f"【推理报错】{error_location} -> {type(e).__name__}: {str(e)}"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
