

from temporis.iterators.iterators import WindowedDatasetIterator
import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.iterators.batcher import Batcher
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


class TestIterators():
    def test_iterators(self):
        features = ['feature1', 'feature2']
        x = ByNameFeatureSelector(features)
        x = PandasMinMaxScaler((-1, 1))(x)

        y = ByNameFeatureSelector(['RUL'])
        transformer = Transformer(x, y)
        batch_size = 15
        window_size = 5
        ds = MockDataset(5)
        transformer.fit(ds)
        b = Batcher.new(ds, window_size, batch_size,
                        transformer, 1, restart_at_end=False)
        X, y, w = next(b)
        assert len(y.ravel()) == batch_size
        assert X.shape[0] == batch_size
        assert X.shape[1] == window_size
        assert X.shape[2] == 2

    def test_2(self):
        dataset = SimpleDataset()
        pipe = ByNameFeatureSelector(['feature1'])
        y_pipe = ByNameFeatureSelector(['RUL'])
        transformer_raw = Transformer(
            transformerX=pipe,    
            transformerY=y_pipe
        )
        transformer_raw.fit(dataset)
        it  = WindowedDatasetIterator(dataset, 5, transformer_raw)
        X, y, sw = it[0]
        assert np.all(X == np.array([[0,1,2,3,4]]).T)
        assert y[0][0] == 4


        X, y, sw = it[-1]
        assert np.all(X == np.array([[95,96,97,98,99]]).T)
        assert y[0][0] == 99
        
        
