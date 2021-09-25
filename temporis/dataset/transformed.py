
from typing import Optional

import pandas as pd
from sklearn.utils.validation import check_is_fitted
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.transformation.functional.transformers import Transformer
from temporis.utils.lrucache import LRUDataCache


class TransformedDataset(AbstractTimeSeriesDataset):
    def __init__(self, dataset, transformer:Transformer, cache_size:Optional[int]=None):
        self.transformer = transformer 
        self.dataset = dataset
        if cache_size is None:
            cache_size = len(dataset)
        self.cache = LRUDataCache(cache_size)
        check_is_fitted(transformer)


    @property
    def n_time_series(self):
        """Base class of the dataset handled by this library.

            Methods for fitting and transform receives an instance
            that inherit from this class
        """
        return self.dataset.n_time_series

    def __call__(self, i:int):
        return self[i]

    def number_of_samples_of_time_series(self, i:int) -> int:
        _, y, _ = self[i]
        return y.shape[0]

        
    def get_time_series(self, i: int) -> pd.DataFrame:
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        if i not in self.cache.data:
            data = self.dataset[i]
            X, y, metadata = self.transformer.transform(data)
            self.cache.add(i, (X.values, y.values, metadata))
        return self.cache.get(i)
    