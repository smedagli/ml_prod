"""
POST: create data
GET: read data
PUT: update data
DELETE: delete data
"""
# Put the code for your API here.
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Union
import os

import sys
sys.path.append('../')
from starter.ml import data, model, encoders

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Load model
model_ = model.Model(verbose=False)
model_.load()

# Load encoder
encoder = encoders.Encoder(verbose=False)
encoder.load()

cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race",
                "sex", "native-country"]  # define categorical features
cat_features = list(map(data.normalize_text, cat_features))

app = FastAPI()


class Example(BaseModel):
    workclass: str = Field(..., example='state_gov')
    education: str = Field(..., example="bachelors")
    marital_status: str = Field(..., example="Never_married")
    occupation: str = Field(..., example="adm_clerical")
    relationship: str = Field(..., example="not_in_family")
    race: str = Field(..., example="white")
    sex: str = Field(..., example="male")
    native_country: str = Field(..., example="united_states")
    age: int = Field(..., example=32)
    education_num: int = Field(..., example=13)
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    fnlgt: int = Field(..., example=77516)
    hours_per_week: int = Field(..., example=40)


@app.post('/predict')
async def predict(input_sample: Example, age: Union[int, None] = None, education_num: Union[int, None] = None):
    data_ = input_sample.dict()
    if age:
        data_.update({'age': age})
    if education_num:
        data_.update({'education_num': education_num})
    df = pd.DataFrame(data_, index=[0])
    x, _ = data.process_data(dataframe=df, categorical_features=cat_features, label=None, encoders=encoder)
    pred = model_.predict(x)
    return {'prediction': str(encoder.label_encoder.inverse_transform(pred)[0])}


@app.get("/")
async def home():
    return {"home": "This is the home; to run inference, insert URL 'http://127.0.0.1:8000/predict"}
