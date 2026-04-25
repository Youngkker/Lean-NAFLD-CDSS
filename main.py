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

# 尝试导入 torch，如果是深度学习模型可能需要它来处理张量
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
    try:
        # 1. 准备数据
        input_data = patient.model_dump()
        input_df = pd.DataFrame([input_data])[features]
        
        # 2. 预处理
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        
        # 3. 智能预测推理引擎 (自动适配不同类型的模型)
        probability = None
        
        # 方案 A: 传统的 sklearn predict_proba 方法
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_std)[0][1]
            
        # 方案 B: TabICL 自定义 predict 方法
        elif hasattr(model, "predict"):
            # 有些模型 predict 直接输出概率，有些输出类别
            res = model.predict(X_std)
            # 如果输出是概率 (浮点数)
            if isinstance(res, (float, np.floating)) or (isinstance(res, np.ndarray) and res.dtype.kind == 'f'):
                probability = res[0][1] if len(np.shape(res)) > 1 else res[0]
            else:
                 raise ValueError("模型预测输出了类别而不是概率，需要调整获取概率的方法。")

        # 方案 C: 深度学习直接前向传播
        elif callable(model): 
             # 对于 PyTorch 模型，通常需要传入 Tensor
             if HAS_TORCH:
                 tensor_input = torch.tensor(X_std, dtype=torch.float32)
                 with torch.no_grad():
                     output = model(tensor_input)
                     
                 # 假设输出需要经过 softmax 转换
                 if hasattr(torch.nn.functional, 'softmax'):
                     prob_tensor = torch.nn.functional.softmax(output, dim=1)
                     probability = prob_tensor[0][1].item()
                 else:
                     probability = output[0][1].item() # 如果模型直接输出概率
             else:
                 # 如果没装 torch，尝试直接传 numpy
                 output = model(X_std)
                 probability = output[0][1] if len(np.shape(output)) > 1 else output[0]
        else:
            raise ValueError("无法找到模型的预测方法 (predict_proba, predict 或直接调用)!")

        if probability is None:
             raise ValueError("无法从模型输出中提取有效概率值。")

        return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
        
    except Exception as e:
        error_type = type(e).__name__
        error_detail = str(e)
        return {"prediction": {"risk_probability": f"【后台报错】{error_type}: {error_detail}"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
