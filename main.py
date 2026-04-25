import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# 尝试导入 torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
    
    # 深度学习模型需进入评估模式
    if HAS_TORCH and hasattr(model, 'eval'):
        model.eval()
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
        
        # 3. 终极推理引擎
        if HAS_TORCH:
            # 深度学习大模型需要 3D 张量输入: (Batch, Sequence, Features)
            tensor_input = torch.tensor(X_std, dtype=torch.float32).unsqueeze(0)
            
            # 🔥 专属 TabICL 模型的特别通道
            if type(model).__name__ == 'TabICL' or hasattr(model, 'forward_with_cache'):
                with torch.no_grad():
                    # 检查模型是否保存了训练缓存 (In-Context Learning 需要上下文)
                    if getattr(model, "has_cache", False):
                        # return_logits=False 直接输出百分比概率
                        output = model.forward_with_cache(X_test=tensor_input, use_cache=True, return_logits=False)
                    else:
                        # 如果没有缓存，我们给它伪造一个空的 y_train 占位符绕过报错
                        y_train_dummy = torch.empty((1, 0), dtype=torch.long)
                        output = model(X=tensor_input, y_train=y_train_dummy, return_logits=False)
                
                # TabICL 的输出格式通常是 (Batch, Test_Size, Classes) -> (1, 1, 2)
                probability = output[0, 0, 1].item()
                return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
            
            # 兼容其他普通的 Torch 模型
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
        error_type = type(e).__name__
        error_detail = str(e)
        return {"prediction": {"risk_probability": f"【后台报错】{error_type}: {error_detail}"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
