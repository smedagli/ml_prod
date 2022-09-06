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


if __name__ == '__main__':
    unittest.main()
