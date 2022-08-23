"""
Script to train machine learning model.
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from typing import List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import pickle


# TODO: remove this when complete
pd.options.display.width = 2500
pd.options.display.max_columns = 25


class Model:
    def __init__(self):
        self.model = LogisticRegression()

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as output_file:
            pickle.dump(obj=self.model, file=output_file)

    def load(self, filename: str) -> None:
        with open(filename, 'rb') as input_file:
            self.model = pickle.load(input_file)

    def train(self, x: np.array, y: np.array) -> None:
        self.model.fit(X=x, y=y)

    def predict(self, x: np.array) -> np.array:
        return self.model.predict(x)


def normalize_text(text: str) -> str:
    """ Normalize any text.

    Args:
        text:

    Returns:

    """
    # return text.strip().lower().replace("-", "_")
    return text.strip().lower()


def read_data():
    df = pd.read_csv('starter/data/census.csv')
    df = df.rename(columns=normalize_text)  # normalize column names
    normalized_text = df.select_dtypes(object).apply(lambda x: x.str.lower().str.strip().str.replace("-", "_"))
    df[df.select_dtypes(object).columns] = normalized_text
    return df


def process_data(dataframe: pd.DataFrame, *, categorical_features: List[str], label: str, training: bool):
    """

    Args:
        dataframe:
        categorical_features:
        label:
        training:

    Returns:

    """
    output_folder = Path('starter/model')

    output_df = dataframe.copy()
    encoders = {}
    for feature in categorical_features + [label]:
        temp_encoder = LabelEncoder()
        output_df[feature] = temp_encoder.fit_transform(dataframe[feature])
        encoders.update({feature: temp_encoder})

        # save the encoders
        output_file = output_folder / f'enc_{feature}.pkl'
        with open(output_file, 'wb') as encoder_file:
            pickle.dump(temp_encoder, encoder_file)

    x = output_df[categorical_features].to_numpy()
    y = output_df[label].to_numpy()

    if training:
        model_file = str((output_folder / 'model.pkl').resolve())
        model = Model()
        model.train(x, y)
        model.save(model_file)
    return x, y, encoders, ''


def get_encoder_dict(encoder_: LabelEncoder) -> Dict[str, int]:
    return {key: value for key, value in zip(encoder_.classes_, encoder_.transform(encoder_.classes_))}


if __name__ == '__main__':
    data = read_data()

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        ]

    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features,
                                                 label="salary", training=True)

    # Proces the test data with the process_data function.

    # Train and save a model.
