# Put the code for your API here.
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import sys
sys.path.append('../')
from ml_prod.starter import common
from ml_prod.starter.ml import data, model


app = FastAPI()


class Example(BaseModel):
    age: int = Field(..., example=32)
    workclass: str = Field(..., example='state_gov')
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never_married")
    occupation: str = Field(..., example="adm_clerical")
    relationship: str = Field(..., example="not_in_family")
    race: str = Field(..., example="white")
    sex: str = Field(..., example="male")
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="united_states")


@app.get("/")
async def home():
    return {"home": "This is the home; to run inference, insert URL 'http://127.0.0.1:8000/inference"}


@app.post("/row_inference/")
async def row_inference(example_input: Example):
    preds = row_inference(example_input).tolist()
    return {'predictions': preds}


@app.post("/inference")
async def inference():
    model_ = model.Model()
    model_.load(common.path_model / 'model.pkl')

    data_ = pd.read_csv(common.path_dataset)
    data_ = data_.rename(columns=lambda x: data.normalize_text(x))
    preds = []
    for n_row, row in enumerate(data_.iterrows()):
        preds.append(test_df(pd.DataFrame(row[1]).T, model_)[0])
        if n_row > 10:
            break

    return {'predictions': preds}


def test_df(dataframe: pd.DataFrame, model_: model.Model()):

    cat_columns = ["workclass", "education", "marital-status",
                   "occupation", "relationship", "race", "sex",
                   "native-country"]  # define categorical features
    cat_columns = list(map(data.normalize_text, cat_columns))

    encoders = data.load_encoders_ohe(skip_label=True)
    cat_features = encoders['categorical'].transform(dataframe[cat_columns])
    if 'salary' in dataframe.columns:
        cols_to_drop = cat_columns + ['salary']
    else:
        cols_to_drop = cat_columns
    numeric_features = dataframe.drop(cols_to_drop, axis=1).values
    features = np.concatenate([cat_features, numeric_features], axis=1)

    pred = model_.predict(features)
    return pred


def row_inference(row_data: Example):
    model_ = model.Model()
    model_.load(common.path_model / 'model.pkl')

    as_dict = row_data.__dict__
    input_data = pd.DataFrame(as_dict, index=[0])
    return test_df(input_data, model_)
