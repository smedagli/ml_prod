import unittest

from ml_prod.starter.ml import data as data_mod


class TestData(unittest.TestCase):
    def test_column_names(self):
        self.assertTrue(data_mod.read_data().columns.tolist()[: 3] == ['age', 'workclass', 'fnlgt'])

    def test_normalize_text(self):
        self.assertTrue(data_mod.normalize_text('  sTa T-us') == 'sta t_us')
        self.assertTrue(data_mod.normalize_text('AAAAAAA') == 'AAAAAAA'.lower())

    def test_read_data(self):
        data_ = data_mod.read_data()
        self.assertTrue(data_.columns.tolist()[: 4] == ['age', 'workclass', 'fnlgt', 'education'])
        self.assertEqual(data_.iloc[0]['age'], 39)

    # def test_process_data(self):
    #     values = ['a', 'b', 'c', 'd', 'e']
    #     labels = ['yes', 'no', 'no', 'yes', 'yep']
    #     data_ = pd.DataFrame({'values': values, 'labels': labels})
    #
    #     encoder_values = LabelEncoder()
    #     encoder_values.fit(data_['values'])
    #     encoder_labels = LabelEncoder()
    #     encoder_labels.fit(data_['labels'])
    #     encoder_dict = {'values': encoder_values, 'labels': encoder_labels}
    #
    #     x_encoded, y_encoded = data_mod.process_data(data_, categorical_features=['values'], label='labels',
    #                                                  encoder_dict_=encoder_dict)
    #     self.assertEqual(list(map(lambda x: x[0], x_encoded)), [0, 1, 2, 3, 4])
    #     self.assertEqual(list(y_encoded), [2, 0, 0, 2, 1])


if __name__ == '__main__':
    unittest.main()
