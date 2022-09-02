"""
Script to train machine learning model.
"""
from ml_prod.starter.ml.performance import summary_slice
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List, Dict
from pathlib import Path
import pandas as pd
import pickle

from ml_prod.starter import common
from ml_prod.starter.ml.model import Model

# TODO: remove this when complete
pd.options.display.width = 2500
pd.options.display.max_columns = 25


def normalize_text(text: str) -> str:
    """ Normalize any text.

    Args:
        text:

    Returns:

    """
    return text.strip().lower().replace("-", "_")
    # return text.strip().lower()


def read_data(filename: str = common.path_dataset):
    """

    Returns:

    """
    df = pd.read_csv(filename)
    df = df.rename(columns=normalize_text)  # normalize column names
    normalized_text = df.select_dtypes(object).apply(lambda x: x.str.lower().str.strip().str.replace("-", "_"))
    df[df.select_dtypes(object).columns] = normalized_text
    return df


def process_data(dataframe: pd.DataFrame, *, categorical_features: List[str], label: str,
                 encoder_dict_: Dict[str, LabelEncoder]) -> pd.DataFrame:
    """ Returns the input dataframe with categorical features and label encoded with the LabelEncoder defined in input.

    Args:
        dataframe: input dataset
        categorical_features: columns with the features
        label: target column
        encoder_dict_: dictionary with keys the name of the columns to encode (features and label) and values the
                      corresponding encoder

    Returns:
        encoded dataframe of features and label

    """
    output_df = dataframe.copy()
    for feature_ in categorical_features + [label]:
        encoder_ = encoder_dict_[feature_]
        # temp_encoder = LabelEncoder()
        # output_df[feature] = temp_encoder.fit_transform(dataframe[feature])
        output_df[feature_] = encoder_.transform(dataframe[feature_])

    return output_df


def get_encoder_dict(encoder_: LabelEncoder) -> Dict[str, int]:
    return {key: value for key, value in zip(encoder_.classes_, encoder_.transform(encoder_.classes_))}


def get_encoder_dict_inv(encoder_: LabelEncoder) -> Dict[int, str]:
    return {key: value for value, key in zip(encoder_.classes_, encoder_.transform(encoder_.classes_))}


def set_encoders(dataframe: pd.DataFrame, categorical_features: List[str], label: str,
                 save_encoders: bool = False) -> Dict[str, LabelEncoder]:
    """

    Args:
        dataframe:
        categorical_features:
        label:
        save_encoders: if True, save the encoders to a .pkl file

    Returns:

    """
    output_folder_ = common.path_model
    out_encoders = {}
    for feature_ in categorical_features + [label]:
        temp_encoder = LabelEncoder()
        temp_encoder.fit(dataframe[feature_])
        out_encoders.update({feature_: temp_encoder})

        if save_encoders:
            output_file = output_folder_ / f'enc_{feature_}.pkl'
            print(f"Saving encoder for {feature_} in {output_file}")
            with open(output_file, 'wb') as encoder_file:
                pickle.dump(temp_encoder, encoder_file)
    return out_encoders


_DO_TRAINING = True


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
    label_column = 'salary'
    cat_features = list(map(normalize_text, cat_features))

    encoders = set_encoders(dataframe=train, categorical_features=cat_features, label=label_column, save_encoders=True)
    train_encoded = process_data(train, categorical_features=cat_features, label=label_column,
                                 encoder_dict_=encoders)

    X_train = train_encoded[cat_features].to_numpy()
    y_train = train_encoded[label_column].to_numpy()

    if not _DO_TRAINING:
        exit()

    output_folder = Path(common.path_model)
    model_file = str((output_folder / 'model.pkl').resolve())
    model = Model()
    model.train(X_train, y_train)
    model.save(model_file)

    pred = model.predict(X_train)
    train_encoded['pred'] = pred
    train['pred'] = pred
    train['label_encoded'] = train_encoded[label_column]

    cm = common.BinaryConfusionMatrix(common.confusion_matrix_df(train_encoded, label_column, 'pred'))
    print(cm.confusion_matrix)
    print(cm.to_df())

    for category in cat_features:
        print(category)
        print(summary_slice(train, category, 'label_encoded', 'pred'))
        print('\n')
