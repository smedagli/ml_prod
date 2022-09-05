"""
Tools and functions to pre-process data
TODO:
    - add implementation of "process_data" as proposed by Udacity
"""
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder

from ml_prod.starter import common
from ml_prod.starter.ml.encoders import Encoder

EncoderType = Union[LabelEncoder, LabelBinarizer, OneHotEncoder]


def normalize_text(text: str) -> str:
    """ Normalize any text.

    Args:
        text:

    Returns:

    """
    return text.strip().lower().replace("-", "_")
    # return text.strip().lower()


def read_data(filename: str = common.path_dataset):
    """ Reads the csv file and "normalizes" the column names.
    Normalizes also the text in each column that is of object type.

    Returns:

    """
    df = pd.read_csv(filename)
    df = df.rename(columns=normalize_text)  # normalize column names
    normalized_text = df.select_dtypes(object).apply(lambda x: x.str.lower().str.strip().str.replace("-", "_"))
    df[df.select_dtypes(object).columns] = normalized_text
    return df


def process_data(dataframe: pd.DataFrame, categorical_features: List[str], label: str,
                 encoders: Encoder) -> Tuple[np.array, np.array]:
    """ Returns the input dataframe with categorical features and label encoded with the LabelEncoder defined in input.

    Args:
        dataframe: input dataset
        categorical_features: columns with the features
        label: target column
        encoders:

    Returns:
        encoded dataframe of features and label

    """
    input_dataframe = dataframe.copy()
    if label in input_dataframe.columns:
        y = encoders.label_encoder.transform(input_dataframe.pop(label))
    else:
        y = np.array([None] * len(input_dataframe))

    cat_variables = input_dataframe[categorical_features].values
    input_dataframe = input_dataframe.drop(categorical_features, axis=1)
    cat_variables_enc = encoders.categorical_features_encoder.transform(cat_variables)

    num_variables = input_dataframe.values
    x = np.concatenate([num_variables, cat_variables_enc], axis=1)
    return x, y
