import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import traceback
import torch  # 🔥 必须引入 torch

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
    
    # 🔥 深度学习模型必须进入评估模式
    if hasattr(model, 'eval'):
        model.eval()
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
        # 1. 准备数据 (确保列顺序与训练时一致)
        input_data = patient.model_dump()
        input_df = pd.DataFrame([input_data])[features]
        
        # 2. 预处理
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        
        # 3. 🔥 深度学习适配：转为 Torch 张量并调整维度
        # 假设模型需要 (1, 1, Features) 的输入格式
        x_tensor = torch.from_numpy(X_std).float()
        if len(x_tensor.shape) == 2:
            x_tensor = x_tensor.unsqueeze(1) # 变成 (1, 1, Features)
            
        # 4. 执行预测
        with torch.no_grad():
            # TabICL 通常需要一个空的 train_label 占位，或者直接 forward
            # 这里我们尝试最通用的深度学习调用方式
            output = model(x_tensor)
            
            # 如果输出是 logits，转为概率
            if torch.max(output) > 1 or torch.min(output) < 0:
                prob = torch.softmax(output, dim=-1)
            else:
                prob = output
            
            # 取出概率值 (假设是二分类，取索引 1)
            risk_val = prob.flatten()[1].item()
        
        return {"prediction": {"risk_probability": f"{risk_val * 100:.1f}"}}
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"\n⚠️ 推理崩溃详情:\n{error_msg}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
