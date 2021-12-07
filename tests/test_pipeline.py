from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.transformation import Concatenate as TransformationConcatenate
from temporis.transformation import Transformer
from temporis.transformation.features.imputers import PerColumnImputer
from temporis.transformation.features.outliers import IQROutlierRemover
from temporis.transformation.features.scalers import MinMaxScaler
from temporis.transformation.features.selection import ByNameFeatureSelector
from temporis.transformation.features.transformation import MeanCentering
from temporis.transformation.functional.graph_utils import root_nodes
from temporis.transformation.functional.pipeline import TemporisPipeline, make_pipeline
from temporis.transformation.functional.transformerstep import TransformerStep
from copy import deepcopy



class Joiner(TransformerStep):
    def transform(self, X: List[Tuple[Any, pd.DataFrame]]):
        print(X)
        return pd.concat([data for category, data in X]).sort_index()
            


class Selector(TransformerStep):
    def __init__(self, category:str,
        category_feature:str,
        name: Optional[str] = None,):
        super().__init__(name)
        self.category = category
        self.category_feature = category_feature

    def transform(self, X):
        return X[X[self.category_feature] == self.category].drop(columns=[self.category_feature])


class SplitByCategory(TransformerStep):
    def __init__(
        self,
        categorical_feature_name: str,
        pipeline : TemporisPipeline,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.orig_pipeline = deepcopy(pipeline)
        self.categorical_feature_name = categorical_feature_name
        self._sub_pipelines = {}
        self.joiner = Joiner()(self)

    def _build_pipeline(self, category):
        self.disconnect(self.joiner)
        s = Selector(category, self.categorical_feature_name)(self)
        new_pipe = deepcopy(self.orig_pipeline)        
        for r in root_nodes(new_pipe):
            r(s)
        self.joiner(new_pipe.final_step)
        return new_pipe

    def transform(self, X):
        return X
   


    def partial_fit(self, X, y=None):

        #if 'default' not in self._sub_pipelines:
        #    self._sub_pipelines['default'] = self._build_pipeline('ALL')
        self.categorical_feature_name = self.find_feature(
            X, self.categorical_feature_name
        )
        for c in X[self.categorical_feature_name].unique():
            if c not in self._sub_pipelines:      
                self._sub_pipelines[c] = self._build_pipeline(c)
        return self
    
    def __call__(
        self, prev: Union[List["TransformerStepMixin"], "TransformerStepMixin"]
    ):
        self._add_previous(prev)
        return self.joiner
                
            


class MockDatasetCategorical(AbstractTimeSeriesDataset):
    def __init__(self):
        super().__init__()
        self.lives = [
            pd.DataFrame(
                {
                    "Categorical": ["a", "a", "b", "b"],
                    "feature1": [500, 515, -45, -66],
                    "feature2": [93, 95, 1500, np.nan],
                }
            ),
            pd.DataFrame(
                {
                    "Categorical": ["a", "a", "b", "b"],
                    "feature1": [499, 525, -45, -66],
                    "feature2": [93, 95, 1500, 1200],
                }
            ),
            pd.DataFrame(
                {
                    "Categorical": ["a", "a", "b", "b"],
                    "feature1": [555500, 5455515, -45, -66],
                    "feature2": [93, 95, 1500, 1200],
                }
            ),
            pd.DataFrame(
                {
                    "Categorical": ["a", "a", "b", "b", "a", "a"],
                    "feature1": [500, 515, -45, -66, 495, 498],
                    "feature2": [93, 95, 1500, 1320, 85, np.nan],
                }
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
        pipe = MinMaxScaler((-1, 1), name="Scaler")(pipe)

        target_pipe = ByNameFeatureSelector(["RUL"])

        test_transformer = Transformer(transformerX=pipe, transformerY=target_pipe)

        test_transformer.fit(dataset)

        X, y, sw = test_transformer.transform(dataset[0])

        assert X.shape[1] == 2
        df_dataset = dataset.to_pandas()

        centered_df = df_dataset[["a", "b"]] - df_dataset[["a", "b"]].mean()
        scaler = test_transformer.transformerX.find_node("Scaler")
        assert scaler.data_min.equals(centered_df.min(axis=0))
        assert scaler.data_max.equals(centered_df.max(axis=0))

    def test_FitOrder2(self):
        dataset = MockDataset()

        pipe_a = ByNameFeatureSelector(["a"])
        pipe_a = MeanCentering()(pipe_a)
        scaler_a = MinMaxScaler((-1, 1), name="a")
        pipe_a = scaler_a(pipe_a)

        pipe_b = ByNameFeatureSelector(["b"])
        pipe_b = MeanCentering()(pipe_b)
        scaler_b = MinMaxScaler((-1, 1), name="b")
        pipe_b = scaler_b(pipe_b)

        pipe = TransformationConcatenate()([pipe_a, pipe_b])

        target_pipe = ByNameFeatureSelector(["RUL"])

        test_transformer = Transformer(transformerX=pipe, transformerY=target_pipe)

        test_transformer.fit(dataset)

        X, y, sw = test_transformer.transform(dataset[0])

        assert X.shape[1] == 2
        df_dataset = dataset.to_pandas()
        centered_df = df_dataset[["a", "b"]] - df_dataset[["a", "b"]].mean()

        assert scaler_a.data_min.equals(centered_df.min(axis=0)[["a"]])
        assert scaler_b.data_max.equals(centered_df.max(axis=0)[["b"]])

    def test_PandasConcatenate(self):
        dataset = MockDataset1()

        pipe = ByNameFeatureSelector(["a"])
        pipe = MinMaxScaler((-1, 1))(pipe)

        pipe2 = ByNameFeatureSelector(["b"])
        pipe2 = MinMaxScaler((-5, 0))(pipe2)

        pipe = TransformationConcatenate()([pipe, pipe2])
        pipe = MeanCentering()(pipe)

        target_pipe = ByNameFeatureSelector(["RUL"])

        test_transformer = Transformer(transformerX=pipe, transformerY=target_pipe)

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

    def test_subpipeline(self):
        dataset = MockDatasetCategorical()
        pipe = ByNameFeatureSelector(["Categorical", "feature1", "feature2"])
        bb = make_pipeline(
            IQROutlierRemover(lower_quantile=0.05, upper_quantile=0.95),
            MinMaxScaler((-1, 1)),
            PerColumnImputer(),
        )
        pipe = SplitByCategory("Categorical", bb)(pipe)
   

        target_pipe = ByNameFeatureSelector(["RUL"])

        test_transformer = Transformer(transformerX=pipe)
        test_transformer.fit(dataset)
