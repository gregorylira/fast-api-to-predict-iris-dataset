import joblib
import tensorflow as tf
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import numpy as np


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

loaded_model.compile(loss='categorical_crossentropy',
                     metrics=['accuracy'], optimizer='rmsprop')


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
