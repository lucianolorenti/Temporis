from typing import Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.iterators.iterators import NotWeighted, SampleWeight, WindowedDatasetIterator
from temporis.iterators.shufflers import AbstractShuffler, NotShuffled

class EstimatorWrapper(TransformerMixin, BaseEstimator):
    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, Xy : Tuple[np.ndarray, np.ndarray], y=None, **fit_params):
        X, y = Xy
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, Xy, **transform_params):
        X, y = Xy
        return self.estimator.predict(X, **transform_params)


class SKLearnTimeSeriesWindowTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, transformer_builder, 
                 window_size: int,
                 step: int = 1,
                 output_size: int = 1,
                 shuffler: AbstractShuffler = NotShuffled(),
                 sample_weight:  SampleWeight= NotWeighted(),
                 add_last: bool = True):
        self.transformer = transformer_builder()
        self.window_size=window_size
        self.output_size = output_size
        self.step = step
        self.shuffler = shuffler 
        self.sample_weight = sample_weight
        self.add_last = add_last

    def fit(self, dataset: AbstractTimeSeriesDataset):
        self.transformer.fit(dataset)
        return self
       

    def transform(self, dataset: AbstractTimeSeriesDataset):
        iterator = WindowedDatasetIterator(
            dataset.map(self.transformer),
            self.window_size,
            self.step,
            self.output_size,
            self.shuffler,self.sample_weight,self.add_last
        )
        X, y, sw = iterator.get_data()
        return X, y

    def true_values(self, dataset: AbstractTimeSeriesDataset):
        iterator = WindowedDatasetIterator(
            dataset.map(self.transformer),
            self.window_size,
            self.step,
            self.output_size,
            self.shuffler,self.sample_weight,self.add_last
        )
        X, y, sw = iterator.get_data()
        return y
