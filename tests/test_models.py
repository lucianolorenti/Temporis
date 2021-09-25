

from temporis.iterators.iterators import WindowedDatasetIterator
import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.iterators.batcher import Batcher
from temporis.models.keras import tf_regression_dataset
from temporis.transformation.features.scalers import PandasMinMaxScaler
from temporis.transformation.features.selection import ByNameFeatureSelector
from temporis.transformation import (TemporisPipeline, Transformer)


class SimpleDataset(AbstractTimeSeriesDataset):
    def __init__(self):

        self.lives = [
            pd.DataFrame({
                'feature1': np.array(range(0, 100)),
                'RUL': np.array(range(0, 100))
            })]


    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return 'RUL'

    @property
    def n_time_series(self):
        return len(self.lives)


class MockDataset(AbstractTimeSeriesDataset):
    def __init__(self, nlives: int):

        self.lives = [
            pd.DataFrame({
                'feature1': np.linspace(0, (i+1)*100, 50),
                'feature2': np.linspace(-25, (i+1)*500, 50),
                'RUL': np.linspace(100, 0, 50)
            })
            for i in range(nlives-1)]

        self.lives.append(
            pd.DataFrame({
                'feature1': np.linspace(0, 5*100, 50),
                'feature2': np.linspace(-25, 5*500, 50),
                'feature3': np.linspace(-25, 5*500, 50),
                'RUL': np.linspace(100, 0, 50)
            })
        )

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return 'RUL'

    @property
    def n_time_series(self):
        return len(self.lives)


class TestModels():
    def test_models(self):
        features = ['feature1', 'feature2']
        x = ByNameFeatureSelector(features)
        x = PandasMinMaxScaler((-1, 1))(x)

        y = ByNameFeatureSelector(['RUL'])
        transformer = Transformer(x, y)
        batch_size = 15
        window_size = 5
        ds = MockDataset(5)
        transformer.fit(ds)
        iterator = WindowedDatasetIterator(
            ds.map(transformer),
            window_size,
            step=1,
            output_size=1
        )

    
        b1 = tf_regression_dataset(iterator).batch(15)
        assert b1.take(1)


        

