
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI()
model = joblib.load('generated/model.pkl')
features = joblib.load('generated/features.pkl')


class VideoFeatures(BaseModel):
    duration_secs: int
    day_of_week: int
    hour_posted: int
    days_live: float
    title_length: int
    
@app.get('/')
def root():
    return {'message': 'YouTube Shorts Like Ratio Predictor API', 'usage': 'POST /predict'}

@app.post('/predict')
def predict(video: VideoFeatures): 
    vals = [[getattr(video, f) for f in features]]
    pred = model.predict(vals)[0]
    return {'predicted_like_ratio': round(float(pred), 4), 'as_percent': f'{pred*100:.1f}%'}
