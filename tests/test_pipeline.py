import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.transformation.features.scalers import MinMaxScaler
from temporis.transformation.features.selection import ByNameFeatureSelector
from temporis.transformation.features.transformation import MeanCentering
from temporis.transformation import Transformer
from temporis.transformation import (
    Concatenate as TransformationConcatenate,
)


class MockDataset(AbstractTimeSeriesDataset):
    def __init__(self):
        super().__init__()
        self.lives = [
            pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "RUL": [4, 3, 2, 1]}),
            pd.DataFrame(
                {"a": [150, 5, 14, 24], "b": [-52, -14, -36, 8], "RUL": [4, 3, 2, 1]}
            ),
        ]

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def n_time_series(self):
        return len(self.lives)


class MockDataset1(AbstractTimeSeriesDataset):
    def __init__(self):
        super().__init__()
        self.lives = [
            pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "RUL": [4, 3, 2, 1]}),
            pd.DataFrame({"a": [2, 4, 6, 8], "b": [2, 4, 6, 8], "RUL": [4, 3, 2, 1]}),
        ]

    def get_time_series(self, i: int):
        return self.lives[i]

    @property
    def rul_column(self):
        return "RUL"

    @property
    def n_time_series(self):
        return len(self.lives)


class TestPipeline:
    def test_FitOrder(self):

        dataset = MockDataset()

        pipe = ByNameFeatureSelector(["a", "b"])
        pipe = MeanCentering()(pipe)
        pipe = MinMaxScaler((-1, 1), name='Scaler')(pipe)

        target_pipe = ByNameFeatureSelector(["RUL"])

        test_transformer = Transformer(transformerX=pipe, transformerY=target_pipe)
        
        test_transformer.fit(dataset)

        X, y, sw = test_transformer.transform(dataset[0])

        assert X.shape[1] == 2
        df_dataset = dataset.to_pandas()

        centered_df = df_dataset[["a", "b"]] - df_dataset[["a", "b"]].mean()
        scaler = test_transformer.transformerX.find_node('Scaler')
        assert scaler.data_min.equals(centered_df.min(axis=0))
        assert scaler.data_max.equals(centered_df.max(axis=0))

    def test_FitOrder2(self):
        dataset = MockDataset()


        pipe_a = ByNameFeatureSelector(["a"])
        pipe_a = MeanCentering()(pipe_a)
        scaler_a =  MinMaxScaler((-1, 1), name="a")
        pipe_a =scaler_a(pipe_a)

        pipe_b = ByNameFeatureSelector(["b"])
        pipe_b = MeanCentering()(pipe_b)
        scaler_b = MinMaxScaler((-1, 1), name="b")
        pipe_b = scaler_b(pipe_b)

        pipe = TransformationConcatenate()([pipe_a, pipe_b])

        target_pipe = ByNameFeatureSelector(["RUL"])

        test_transformer = Transformer(
            transformerX=pipe, transformerY=target_pipe)
    
        test_transformer.fit(dataset)

        X, y, sw = test_transformer.transform(dataset[0])

        assert X.shape[1] == 2
        df_dataset = dataset.to_pandas()
        centered_df = df_dataset[["a", "b"]] - df_dataset[["a", "b"]].mean()

        assert scaler_a.data_min.equals(centered_df.min(axis=0)[['a']])
        assert scaler_b.data_max.equals(centered_df.max(axis=0)[['b']])

    def test_PandasConcatenate(self):
        dataset = MockDataset1()

        pipe = ByNameFeatureSelector(["a"])
        pipe = MinMaxScaler((-1, 1))(pipe)

        pipe2 = ByNameFeatureSelector(["b"])
        pipe2 = MinMaxScaler((-5, 0))(pipe2)

        pipe = TransformationConcatenate()([pipe, pipe2])
        pipe = MeanCentering()(pipe)

        target_pipe = ByNameFeatureSelector(["RUL"])

        test_transformer = Transformer(
            transformerX=pipe, transformerY=target_pipe)

        test_transformer.fit(dataset)

        df = dataset.to_pandas()[["a", "b"]]

        data_min = df.min()
        data_max = df.max()

        gt = (df - data_min) / (data_max - data_min)
        gt["a"] = gt["a"] * (1 - (-1)) + (-1)
        gt["b"] = gt["b"] * (0 - (-5)) + (-5)
        gt = gt - gt.mean()

        X, y, sw = test_transformer.transform(dataset[0])


        assert (np.mean((gt.iloc[:4, :].values - X.values) ** 2)) < 0.0001
        X, y, sw = test_transformer.transform(dataset[1])
        assert (np.mean((gt.iloc[4:, :].values - X.values) ** 2)) < 0.0001

        assert X.shape[1] == 2