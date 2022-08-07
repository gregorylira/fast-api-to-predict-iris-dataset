from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import pickle


# uvicorn main:app --reload

loaded_model = pickle.load(open('LRClassifier.pkl', 'rb'))


app = FastAPI()


Species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


class Predict(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(pred: Predict):
    y_pred = loaded_model.predict(
        [[pred.SepalLengthCm, pred.SepalWidthCm, pred.PetalLengthCm, pred.PetalWidthCm]])
    print(y_pred)
    return {"predict": Species[y_pred[0]]}
