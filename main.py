import tensorflow as tf
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import numpy as np


# uvicorn main:app --reload

loaded_model_json = tf.keras.models.load_model('model.h5')


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
    y_predArg = np.argmax(y_pred, axis=1)
    print(y_predArg)
    return {"predict": Species[y_predArg[0]]}
