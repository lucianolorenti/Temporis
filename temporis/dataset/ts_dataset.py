"""The Dataset module provides a light interface to define a PM Dataset
"""
from collections.abc import Iterable
from re import S

from typing import Any, List, Tuple, Union

import numpy as np
from numpy.lib.arraysetops import isin
import tensorflow as tf
import pandas as pd
from tqdm.auto import tqdm


class AbstractTimeSeriesDataset:
    def __init__(self):
        self._common_features = None
        self._durations = None

    @property
    def n_time_series(self) -> int:
        raise NotImplementedError

    def number_of_samples_of_time_series(self, i: int) -> int:
        return self[i].shape[0]

    def get_time_series(self, i: int) -> pd.DataFrame:
        """

        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        raise NotImplementedError

    def duration(self, life: pd.DataFrame) -> float:
        """Obtain the duration of the time-series

        Parameters
        ----------
        i : int
            Index of the life

        Returns
        -------
        float
            Duration of the life
        """
        v = life.index
        return v.max() - v.min()

    def durations(self) -> List[float]:
        """Obtain the length of each life

        Returns
        -------
        List[float]
            List of durations
        """
        if self._durations is None:
            self._durations = [self.duration(life) for life in self]
            # [self.rul_column].iloc[0]
        return self._durations

    def __call__(self, i):
        return self[i]

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
        if isinstance(i, tf.Tensor):
            return self.get_time_series(i.ref())
        if isinstance(i, Iterable):
            if not all(isinstance(item, (int, np.integer)) for item in i):
                raise ValueError("Invalid iterable index passed")

            return FoldedDataset(self, i)
        else:
            df = self.get_time_series(i)
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

    def to_pandas(
        self,
        proportion_of_lives: float = 1.0,
        subsample_proportion: float = 1.0,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """
        Create a dataset with the time-series concatenated

        Parameters
        ----------
        proportion_of_lives: float
                    Proportion of lives to use.

        Returns
        -------

        pd.DataFrame:
            Return a DataFrame with all the lives concatenated
        """
        if show_progress:
            bar = tqdm
        else:
            bar = lambda x: x
        df = []

        features = list(
            self._compute_common_features(
                proportion_of_lives=proportion_of_lives, show_progress=show_progress
            )
        )

        for i in bar(range(self.n_time_series)):

            if proportion_of_lives < 1.0 and np.random.rand() > proportion_of_lives:
                continue

            current_life = self[i].loc[:, features]
            if subsample_proportion < 1.0:
                indices = range(
                    0,
                    current_life.shape[0],
                    int(current_life.shape[0] * subsample_proportion),
                )
                current_life = current_life.iloc[indices, :]
            df.append(current_life)
        return pd.concat(df)

    def _compute_common_features(
        self, proportion_of_lives: float = 1.0, show_progress: bool = False
    ) -> List[str]:
        common_features = []
        if show_progress:
            bar = tqdm
        else:
            bar = lambda x: x
        for i in bar(range(self.n_time_series)):
            if proportion_of_lives < 1.0 and np.random.rand() > proportion_of_lives:
                continue
            life = self[i]
            common_features.append(set(life.columns.values))
        return common_features[0].intersection(*common_features)

    def common_features(
        self, show_progress: bool = False, proportion_of_lives: float = 1.0
    ) -> List[str]:
        if self._common_features is None:
            self._common_features = self._compute_common_features(
                proportion_of_lives, show_progress=show_progress
            )
        return self._common_features

    def map(self, transformer, cache_size: int = None):
        from temporis.dataset.transformed import TransformedDataset

        return TransformedDataset(self, transformer, cache_size=cache_size)

    def numeric_features(self, show_progress: bool = False) -> List[str]:
        """Obtain the list of the common numeric features in the dataset

        Parameters
        ----------
        show_progress : bool, optional
            Whether to show progress when computing the common features, by default False

        Returns
        -------
        List[str]
            List of columns
        """

        features = self.common_features(show_progress=show_progress)
        df = self.get_time_series(0)
        return list(
            df.loc[:, features]
            .select_dtypes(include=[np.number], exclude=["datetime", "timedelta"])
            .columns.values
        )

    def categorical_features(self, show_progress: bool = False) -> List[str]:
        features = self.common_features(show_progress=show_progress)
        df = self.get_time_series(0)
        return list(
            df.loc[:, features]
            .select_dtypes(exclude=[np.number, "datetime", "timedelta"])
            .columns.values
        )


class FoldedDataset(AbstractTimeSeriesDataset):
    def __init__(self, dataset: AbstractTimeSeriesDataset, indices: list):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    @property
    def n_time_series(self):
        return len(self.indices)

    def get_time_series(self, i: int):
        return self.dataset[self.indices[i]]

    def _original_index(self, i: int):
        if isinstance(self.dataset, FoldedDataset):
            return self.dataset._original_index(self.indices[i])
        else:
            return self.indices[i]

    def original_indices(self):
        return [self._original_index(i) for i in range(len(self.indices))]
