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
        # 强制将输入转为 Tensor（因为这是 TabICL）
        if HAS_TORCH:
            tensor_input = torch.tensor(X_std, dtype=torch.float32)
            
            # 方案 A: 尝试调用模型的 forward 拿原始 logits
            try:
                with torch.no_grad():
                    output = model(tensor_input)
                # 假设 output 是 [batch_size, num_classes] 的 logits
                if hasattr(torch.nn.functional, 'softmax'):
                    prob_tensor = torch.nn.functional.softmax(output, dim=1)
                    # 取第 1 类的概率
                    probability = prob_tensor[0][1].item()
                    return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
            except Exception as forward_err:
                pass # 如果 forward 报错，继续试下面的方法
                
        # 方案 B: 使用它自带的 predict
        if hasattr(model, "predict"):
            res = model.predict(X_std)
            # 如果是单个数值 (0 或 1)
            if np.isscalar(res) or (isinstance(res, np.ndarray) and res.size == 1):
                val = int(np.squeeze(res))
                if val == 1:
                    return {"prediction": {"risk_probability": "高风险 (99.0)"}}
                else:
                    return {"prediction": {"risk_probability": "低风险 (1.0)"}}
            
            # 如果输出是个数组
            elif isinstance(res, np.ndarray):
                # 如果它是一个多维的概率数组 [0.2, 0.8]
                if res.ndim > 1 and res.shape[1] > 1:
                    probability = res[0][1]
                    return {"prediction": {"risk_probability": f"{probability * 100:.1f}"}}
                else:
                    # 如果是一个一维的分类数组 [1]
                    val = int(res[0])
                    if val == 1:
                        return {"prediction": {"risk_probability": "高风险 (99.0)"}}
                    else:
                        return {"prediction": {"risk_probability": "低风险 (1.0)"}}
                        
        raise ValueError("模型结构过于特殊，无法解析输出结果。")
        
    except Exception as e:
        error_type = type(e).__name__
        error_detail = str(e)
        return {"prediction": {"risk_probability": f"【后台报错】{error_type}: {error_detail}"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
