
import gzip
import pickle
from pathlib import Path
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
    def n_time_series(self) -> int:
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

    def save(self, output_path:Path):
        TransformedSerializedDataset.save(self, output_path)


class TransformedSerializedDataset(TransformedDataset):
    @staticmethod
    def save(dataset:TransformedDataset, output_path:Path):
        for i, life in enumerate(dataset):
            with gzip.open(output_path / f'ts_{i}.pkl.gz', 'wb') as file:
                pickle.dump(life, file)
        with open(output_path / 'transformer.pkl.gz', 'wb') as file:
            pickle.dump(dataset.transformer, file)

    def __init__(self, dataset_path:Path, cache_size:Optional[int] = None):
        self.dataset_path = dataset_path
        self.files = list(dataset_path.glob('ts_*.pkl.gz'))
        with open(dataset_path / 'transformer.pkl.gz', 'rb') as file:
            self.transformer = pickle.load(file)
        if cache_size is None:
            cache_size = len(self.files)
        self.cache = LRUDataCache(cache_size)

    def _open_file(self, i:int):
        with gzip.open(self.dataset_path / self.files[i], 'rb') as file:
            return pickle.load(file)

    def get_time_series(self, i: int) -> pd.DataFrame:
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        if i not in self.cache.data:
            X, y, metadata = self._open_file(i)
            self.cache.add(i, (X, y, metadata))
        return self.cache.get(i)

    @property
    def n_time_series(self):
        return len(self.files)
