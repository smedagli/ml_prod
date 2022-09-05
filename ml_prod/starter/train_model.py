"""
Script to train machine learning model.
TODO:
    - Optional enhancement, use K-fold cross validation instead of a train-test split.
"""
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import argparse

from ml_prod.starter import common
from ml_prod.starter.ml.model import Model
from ml_prod.starter.ml.performance import PerformanceEvaluator
from ml_prod.starter.ml import data, encoders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='Do testing (inference) instead of training')
    parser.add_argument('-v', '--verbose', default=1, help='Set level of verbosity (0 to 2)')
    parser.add_argument('-e', '--encoder_type', default='OHE', type=str,
                        help='Encoding for categorical features: OHE (One Hot Encoding) or LE (Label Encoder)')
    return parser.parse_args()


if __name__ == '__main__':
    # args = parse_args()
    # _DO_TRAINING = not args.test
    #
    # # verbosity
    # verbosity = int(args.verbose)
    # _VERBOSE, _VERBOSE_MODEL = False, False
    # if verbosity == 0:
    #     pass
    # elif verbosity == 1:
    #     _VERBOSE_MODEL = True
    # elif verbosity == 2:
    #     _VERBOSE, _VERBOSE_MODEL = True, True
    #
    # _ENCODER_TYPE = args.encoder_type

    # read data
    data_ = data.read_data()

    # data segregation
    train_df, test_df = train_test_split(data_, test_size=0.20, random_state=42)  # split dataset
    datasets = {'train': train_df, 'test': test_df}

    # define categorical features and label
    cat_features = ["workclass", "education", "marital-status",
                    "occupation", "relationship", "race", "sex",
                    "native-country"]  # define categorical features
    label_column = 'salary'  # define label column
    cat_features = list(map(data.normalize_text, cat_features))

    # initialize encoders
    enc = encoders.Encoder()
    enc.label_encoder.fit(data_[label_column])  # encoders on full set of data
    enc.categorical_features_encoder.fit(data_[cat_features])
    enc.save()

    common_args = {'categorical_features': cat_features, 'label': label_column, 'encoders': enc}
    # preprocess (categorical features encoding)
    x, y = {}, {}
    for key in ['train', 'test']:
        x_encoded, y_encoded = data.process_data(datasets[key], **common_args)
        x.update({key: x_encoded})
        y.update({key: y_encoded})

    # load model
    model = Model()

    # train
    model.train(x['train'], y['train'])
    model.save()

    # performance
    pred, cm = {}, {}
    for key in ['train', 'test']:
        predictions = model.predict(x[key])
        pred.update({key: predictions})
        df = datasets[key]
        df['label_encoded'] = y[key]
        df['pred'] = predictions

        cm.update({key: common.BinaryConfusionMatrix(common.confusion_matrix_df(df, 'label_encoded', 'pred'))})
        cm_output_file = str(Path(common.path_module) / 'data' / 'performance' / f'{key}_confusion_matrix.csv')
        pd.DataFrame(cm[key].confusion_matrix).to_csv(cm_output_file)  # save confusion matrix

        evaluator = PerformanceEvaluator(dataframe=df, label_column='label_encoded', prediction_column='pred')
        for category in cat_features:
            output_file = str(Path(common.path_module) / 'data' / 'performance' / f'{key}_{category}_slice.csv')
            evaluator.get_summary_slice(category, output_file)
