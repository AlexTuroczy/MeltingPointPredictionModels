import unittest

from data import utils

import pandas as pd


class TestUtils(unittest.TestCase):

    def test_load(self):
        data_df = utils.load_data()
        self.assertTrue("Tm" in data_df.columns)

    def test_test_load(self):
        data_df = utils.load_test_data()
        self.assertTrue("Tm" not in data_df.columns)

    def test_clean(self):
        data_df = utils.load_data()
        clean_df = utils.clean_for_xgboost(data_df)

        for col in clean_df.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(clean_df[col]))


    def test_split_data(self):
        data_df = utils.load_data()
        x_train, y_train, x_val, y_val = utils.split_data(data_df, rng=0)

        self.assertTrue("Tm" not in x_train.columns)
        self.assertTrue("Tm" not in x_val.columns)

        self.assertTrue(y_train.size == x_train.index.size)
        self.assertTrue(y_val.size == x_val.index.size)