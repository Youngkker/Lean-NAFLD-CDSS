import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# 1. 初始化系统
app = FastAPI(title="瘦型脂肪肝预测模型后端")

# 2. 允许网页跨域访问（必须要有）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 挂载模型（请确保这个文件和您的 pkl 模型文件在同一个大目录下）
MODEL_DIR = "Final_Clinical_Practicality_Arena"
try:
    print("⏳ 正在加载核心武器库...")
    imputer = joblib.load(os.path.join(MODEL_DIR, "Imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "Scaler.pkl"))
    champion_model = joblib.load(os.path.join(MODEL_DIR, "Champion_Model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "Features.pkl"))
    print("✅ 武器库挂载成功！")
except Exception as e:
    print(f"❌ 模型加载失败，请检查路径: {e}")


# 4. 定义输入格式
class PatientData(BaseModel):
    ALT: float
    TG: float
    BMI: float
    UA: float
    Glucose: float


# 5. 预测接口
@app.post("/predict_nafld")
async def predict_nafld(patient: PatientData):
    try:
        input_dict = patient.dict()
        input_df = pd.DataFrame([input_dict], columns=features)
        X_imp = imputer.transform(input_df)
        X_std = scaler.transform(X_imp)
        probability = champion_model.predict_proba(X_std)[0][1]

        return {
            "prediction": {
                "risk_probability": f"{probability * 100:.1f}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔥 核心改动：加上这句，您就可以直接在 PyCharm 点击绿色按钮运行了！
if __name__ == "__main__":
    print("🚀 正在启动 AI 引擎，请不要关闭此窗口...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
