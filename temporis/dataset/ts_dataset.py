"""The Dataset module provides a light interface to define a PM Dataset
"""
from collections.abc import Iterable

from typing import List,  Tuple, Union

import numpy as np

import pandas as pd




class AbstractTimeSeriesDataset:
    def __init__(self):
        self._common_features = None
        self._durations = None

    @property
    def n_time_series(self):
        raise NotImplementedError

    """Base class of the dataset handled by this library.

        Methods for fitting and transform receives an instance
        that inherit from this class
    """

    def get_time_series(self, i: int) -> pd.DataFrame:
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        raise NotImplementedError

    def durations(self) -> List[float]:
        """Obtain the length of each life

        Returns
        -------
        List[float]
            List of durations
        """
        if self._durations is None:
            self._durations = [life[self.rul_column].iloc[0] for life in self]
        return self._durations

    def __getitem__(self, i: Union[int, Iterable]):
        """Obtain a time-series or an splice of the dataset using a FoldedDataset

        Parameters
        ----------
        i: Union[int, Iterable]
           If the parameter is an in it will return a pd.DataFrame with the i-th time-series
           If the parameter is a list of int it will return a FoldedDataset
           with the time-series whose id are present in the list


        Raises
        ------
        ValueError: WHen the list does not contain integer parameters

        Returns:
            pd.DataFrame: the i-th time-series
            FoldedDataset: The dataset with the lives specified by the list
        """
        if isinstance(i, slice):
            i = range(
                0 if i.start is None else i.start,
                len(self) if i.stop is None else i.stop,
                1 if i.step is None else i.step,
            )

        if isinstance(i, Iterable):
            if not all(isinstance(item, (int, np.integer)) for item in i):
                raise ValueError("Invalid iterable index passed")

            return FoldedDataset(self, i)
        else:
            df = self.get_time_series(i)
            df["life"] = i
            return df

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_time_series, 1)

    def __len__(self):
        """
        Return
        ------
        int:
            The number of time-series in the dataset
        """
        return self.n_time_series

    def to_pandas(self, proportion=1.0) -> pd.DataFrame:
        """
        Create a dataset with the time-series concatenated

        Parameters
        ----------
        proportion: float
                    Proportion of lives to use.

        Returns
        -------

        pd.DataFrame:
            Return a DataFrame with all the lives concatenated
        """
        df = []
        common_features = self.common_features()
        for i in range(self.n_time_series):
            if proportion < 1.0 and np.random.rand() > proportion:
                continue
            current_life = self[i][common_features]
            df.append(current_life)
        return pd.concat(df)

    def common_features(self) -> List[str]:
        if self._common_features is None:
            self._common_features = []
            for i in range(self.n_time_series):
                life = self[i]
                self._common_features.append(set(life.columns.values))
            self._common_features = self._common_features[0].intersection(
                *self._common_features
            )
        return self._common_features

    def map(self, transformer):
        from temporis.data.transformed_datastet import TransformedDataset
        return TransformedDataset(self, transformer)


class FoldedDataset(AbstractTimeSeriesDataset):
    def __init__(self, dataset: AbstractTimeSeriesDataset, indices: list):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    @property
    def n_time_series(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        return self.dataset[self.indices[i]]
