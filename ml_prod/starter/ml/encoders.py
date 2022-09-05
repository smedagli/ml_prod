from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pickle
from pathlib import Path

from ml_prod.starter import common


class Encoder:
    def __init__(self, verbose: bool = True):
        self.label_encoder = LabelBinarizer()
        self.categorical_features_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self._verbose = verbose

    def save(self, output_folder: Path = common.path_model) -> None:
        label_out_file = output_folder / 'label_enc.pkl'
        feature_out_file = output_folder / 'cat_enc.pkl'

        if not output_folder.is_dir():
            output_folder.mkdir(parents=True)

        with open(label_out_file, 'wb') as label_file:
            pickle.dump(obj=self.label_encoder, file=label_file)

        with open(feature_out_file, 'wb') as feature_file:
            pickle.dump(obj=self.categorical_features_encoder, file=feature_file)

        if self._verbose:
            print(f"Saved encoders in {label_out_file.resolve()} and {feature_out_file.resolve()}")

    def load(self, input_folder: Path = common.path_model) -> None:
        label_in_file = input_folder / 'label_enc.pkl'
        feature_in_file = input_folder / 'cat_enc.pkl'

        with open(label_in_file, 'rb') as label_file:
            label_enc = pickle.load(label_file)

        with open(feature_in_file, 'rb') as feature_file:
            feature_enc = pickle.load(file=feature_file)

        self.categorical_features_encoder = feature_enc
        self.label_encoder = label_enc

        if self._verbose:
            print(f"Loaded encoders from {label_in_file.resolve()} and {feature_in_file.resolve()}")
