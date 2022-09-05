"""
Script to train machine learning model.
TODO:
    - Optional enhancement, use K-fold cross validation instead of a train-test split.
"""
from sklearn.model_selection import train_test_split
from typing import List
from typing import List, Tuple
from pathlib import Path
import pandas as pd
import argparse
import numpy as np

from ml_prod.starter import common
from ml_prod.starter.ml.model import Model
from ml_prod.starter.ml.performance import PerformanceEvaluator
from ml_prod.starter.ml import data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='Do testing (inference) instead of training')
    parser.add_argument('-v', '--verbose', default=1, help='Set level of verbosity (0 to 2)')
    parser.add_argument('-e', '--encoder_type', default='OHE', type=str,
                        help='Encoding for categorical features: OHE (One Hot Encoding) or LE (Label Encoder)')
    return parser.parse_args()


def train(model: Model, full_data: pd.DataFrame, train_data: pd.DataFrame, categorical_features: List[str], label: str,
          encoder_type: str = 'le') -> pd.DataFrame:
    common_args = {'categorical_features': categorical_features, 'label': label, 'encoder_type': encoder_type}
    encoders = data.set_encoders(dataframe=full_data, save_encoders=True,
                                 **common_args)  # initialize LabelEncoders for features and label

    x_train, y_train = data.process_data(train_data, encoder_dict_=encoders, **common_args)  # encode the input dataset
    # train
    model.train(x_train, y_train)
    model.save(common.path_model / 'model.pkl')
    pred = model.predict(x_train)

    out_dataset = train_data.copy()
    out_dataset['pred'] = pred
    out_dataset['label_encoded'] = y_train
    return out_dataset


def test(model_file: str, test_data: pd.DataFrame, categorical_features: List[str], label: str,
         encoder_type: str = 'le') -> pd.DataFrame:
    common_args = {'categorical_features': categorical_features, 'label': label, 'encoder_type': encoder_type}
    encoder_dict = data.load_encoders(**common_args)

    x_test, y_test = data.process_data(test_data, encoder_dict_=encoder_dict, **common_args)
    # infer
    model = Model()
    model.load(model_file)
    pred = model.predict(x_test)
    return np.array(pred), np.array(y_test)


def test_to_df(model_file: str, test_data: pd.DataFrame, categorical_features: List[str], label: str,
               encoder_type: str = 'le') -> pd.DataFrame:
    """ Returns the predictions as a dataframe """
    pred, y_test = test(model_file, test_data, categorical_features, label, encoder_type)

    out_dataset = test_data.copy()
    out_dataset['pred'] = pred
    out_dataset['label_encoded'] = y_test
    return out_dataset


if __name__ == '__main__':
    args = parse_args()
    _DO_TRAINING = not args.test

    # verbosity
    verbosity = int(args.verbose)
    _VERBOSE, _VERBOSE_MODEL = False, False
    if verbosity == 0:
        pass
    elif verbosity == 1:
        _VERBOSE_MODEL = True
    elif verbosity == 2:
        _VERBOSE, _VERBOSE_MODEL = True, True

    _ENCODER_TYPE = args.encoder_type

    # main
    data_ = data.read_data()

    train_df, test_df = train_test_split(data_, test_size=0.20, random_state=42)  # split dataset

    cat_features = ["workclass", "education", "marital-status",
                    "occupation", "relationship", "race", "sex",
                    "native-country"]  # define categorical features
    label_column = 'salary'  # define label column
    cat_features = list(map(data.normalize_text, cat_features))
    common_arguments = {'categorical_features': cat_features, 'label': label_column, 'encoder_type': _ENCODER_TYPE}

    # initialize model
    output_folder = Path(common.path_model)
    model_file = str((output_folder / 'model.pkl').resolve())
    model_ = Model(verbose=_VERBOSE_MODEL)

    if _DO_TRAINING:  # train
        output_prefix = 'train'  # will be the prefix of output performance files
        dataset = train(model=model_, full_data=data_, train_data=train_df, **common_arguments)
    else:  # test
        output_prefix = 'test'  # will be the prefix of output performance files
        dataset = test(model=model_, test_data=test_df, **common_arguments)

    # performance evaluation
    cm = common.BinaryConfusionMatrix(common.confusion_matrix_df(dataset, 'label_encoded', 'pred'))
    cm_output_file = str(Path(common.path_module) / 'data' / 'performance' / f'{output_prefix}_confusion_matrix.csv')
    pd.DataFrame(cm.confusion_matrix).to_csv(cm_output_file)  # save confusion matrix

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
