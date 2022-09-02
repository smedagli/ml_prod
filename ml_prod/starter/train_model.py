"""
Script to train machine learning model.
TODO:
    - test case of OneHotEncoder
    - Optional enhancement, use K-fold cross validation instead of a train-test split.
"""
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from pathlib import Path
import pandas as pd

from ml_prod.starter import common
from ml_prod.starter.ml.model import Model
from ml_prod.starter.ml.performance import PerformanceEvaluator
from ml_prod.starter.ml import data

# TODO: remove this when complete
pd.options.display.width = 2500
pd.options.display.max_columns = 25

_DO_TRAINING = False
_VERBOSE = True
_ENCODER_TYPE = 'le'


def train(model: Model, full_data: pd.DataFrame, train_data: pd.DataFrame, categorical_features: List[str], label: str,
          encoder_type: str = 'le') -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_args = {'categorical_features': categorical_features, 'label': label}
    encoders = data.set_encoders(dataframe=full_data, save_encoders=True, encoder_type=encoder_type,
                                 **common_args)  # initialize LabelEncoders for features and label

    train_encoded = data.process_data(train_data, encoder_dict_=encoders, **common_args)  # encode the input dataset
    X_train = train_encoded[cat_features].to_numpy()
    y_train = train_encoded[label_column].to_numpy()
    # train
    model.train(X_train, y_train)
    model.save(model_file)
    pred = model.predict(X_train)

    out_dataset = train_data.copy()
    out_dataset_encoded = train_encoded.copy()
    out_dataset['pred'] = pred
    out_dataset['label_encoded'] = out_dataset_encoded[label_column]
    out_dataset_encoded['pred'] = pred

    return out_dataset, out_dataset_encoded


def test(model: Model, test_data: pd.DataFrame, categorical_features: List[str], label: str):
    encoder_dict = data.load_encoders(**common_arguments)

    test_encoded = data.process_data(test_data, categorical_features=categorical_features, label=label,
                                     encoder_dict_=encoder_dict)  # encode the input dataset
    X_test = test_encoded[cat_features].to_numpy()
    # y_test = test_encoded[label_column].to_numpy()
    # infer
    model.load(model_file)
    pred = model.predict(X_test)

    out_dataset = test_data.copy()
    out_dataset_encoded = test_encoded.copy()
    out_dataset['pred'] = pred
    out_dataset['label_encoded'] = out_dataset_encoded[label_column]
    out_dataset_encoded['pred'] = pred
    return out_dataset, out_dataset_encoded


if __name__ == '__main__':
    data_ = data.read_data()

    train_df, test_df = train_test_split(data_, test_size=0.20)  # split dataset

    cat_features = ["workclass", "education", "marital-status",
                    "occupation", "relationship", "race", "sex",
                    "native-country"]  # define categorical features
    label_column = 'salary'  # define label column
    cat_features = list(map(data.normalize_text, cat_features))
    common_arguments = {'categorical_features': cat_features, 'label': label_column}

    # initialize model
    output_folder = Path(common.path_model)
    model_file = str((output_folder / 'model.pkl').resolve())
    model_ = Model(verbose=_VERBOSE)

    if _DO_TRAINING:  # train
        output_prefix = 'train'  # will be the prefix of output performance files
        dataset, dataset_encoded = train(model=model_, full_data=data_, train_data=train_df,
                                         encoder_type=_ENCODER_TYPE, **common_arguments)
    else:  # test
        output_prefix = 'test'  # will be the prefix of output performance files
        dataset, dataset_encoded = test(model=model_, test_data=test_df, **common_arguments)

    # performance evaluation
    cm = common.BinaryConfusionMatrix(common.confusion_matrix_df(dataset_encoded, label_column, 'pred'))
    if _VERBOSE:
        print(cm.confusion_matrix)
        print(cm.to_df())

    evaluator = PerformanceEvaluator(dataframe=dataset, label_column='label_encoded', prediction_column='pred')
    for category in cat_features:
        output_file = str(Path(common.path_module) / 'data' / 'performance' / f'{output_prefix}_{category}_slice.csv')
        if _VERBOSE:
            print(category)
            print(evaluator.get_summary_slice(category, output_file=output_file))
            print('\n')
