from enum import Enum
import logging
import random
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from temporis.dataset.transformed import TransformedDataset
from temporis.iterators.shufflers import AbstractShuffler, NotShuffled


logger = logging.getLogger(__name__)


class AbstractSampleWeights:
    def __call__(self, y, i: int, metadata):
        raise NotImplementedError


class NotWeighted(AbstractSampleWeights):
    def __call__(self, y, i: int, metadata):
        return 1


class RULInverseWeighted(AbstractSampleWeights):
    def __call__(self, y, i: int, metadata):
        return 1 / (y[i, 0] + 1)


class InverseToLengthWeighted(AbstractSampleWeights):
    def __call__(self, y, i: int, metadata):
        return 1 / y[0]


SampleWeight = Union[AbstractSampleWeights, Callable[[np.ndarray, int, Any], float]]





def seq_to_seq_signal_generator(
    signal_X,
    signal_y,
    i: int,
    window_size: int,
    output_size: int = 1,
    add_last: bool = True,
):
    initial = max(i - window_size + 1, 0)
    signal_X_1 = signal_X[initial : i + (1 if add_last else 0), :]
    signal_y_1 = signal_y[initial : i + (1 if add_last else 0), :]
    return (signal_X_1, signal_y_1)


def windowed_signal_generator(
    signal_X,
    signal_y,
    i: int,
    window_size: int,
    output_size: int = 1,
    add_last: bool = True,
):
    """
    Return a lookback window and the value to predict.

    Parameters
    ----------
    signal_X:
             Matrix of size (life_length, n_features) with the information of the life
    signal_y:
             Target feature of size (life_length)
    i: int
       Position of the value to predict

    window_size: int
                 Size of the lookback window

    output_size: int
                 Number of points of the target

    add_last: bool


    Returns
    -------
    tuple (np.array, float)
    """
    initial = max(i - window_size + 1, 0)
    signal_X_1 = signal_X[initial : i + (1 if add_last else 0), :]

    if len(signal_y.shape) == 1:

        signal_y_1 = signal_y[i : min(i + output_size, signal_y.shape[0])]

        if signal_y_1.shape[0] < output_size:
            padding = np.zeros(output_size - signal_y_1.shape[0])
            signal_y_1 = np.hstack((signal_y_1, padding))
        signal_y_1 = np.expand_dims(signal_y_1, axis=1)
    else:
        signal_y_1 = signal_y[i : min(i + output_size, signal_y.shape[0]), :]

        if signal_y_1.shape[0] < output_size:
            padding = np.zeros(
                ((output_size - signal_y_1.shape[0]), signal_y_1.shape[1])
            )
            signal_y_1 = np.concatenate((signal_y_1, padding), axis=0)

    if signal_X_1.shape[0] < window_size:

        signal_X_1 = np.vstack(
            (
                np.zeros((window_size - signal_X_1.shape[0], signal_X_1.shape[1])),
                signal_X_1,
            )
        )

    return (signal_X_1, signal_y_1)



class IterationType(Enum):
    SEQ_TO_SEQ = 1
    FORECAST = 2

class WindowedDatasetIterator:
    def __init__(
        self,
        dataset: TransformedDataset,
        window_size: int,
        step: int = 1,
        output_size: int = 1,
        shuffler: AbstractShuffler = NotShuffled(),
        sample_weight: SampleWeight = NotWeighted(),
        add_last: bool = True,
        padding: bool = False,
        iteration_type:IterationType = IterationType.FORECAST
    ):

        self.dataset = dataset
        self.shuffler = shuffler
        self.window_size = window_size
        self.step = step
        self.shuffler.initialize(self)
        self.iteration_type = iteration_type 

        if self.iteration_type == IterationType.FORECAST:
            self.slicing_function = windowed_signal_generator
        else:
            self.slicing_function = seq_to_seq_signal_generator

        if not isinstance(sample_weight, AbstractSampleWeights) or not callable(
            sample_weight
        ):
            raise ValueError(
                "sample_weight should be an AbstractSampleWeights or a callable"
            )

        self.sample_weight = sample_weight

        self.i = 0
        self.output_size = output_size
        self.add_last = add_last
        self.length = None
        self.padding = padding
        if not self.padding:
            self.valid_sample = lambda x: x >= self.window_size - 1
        else:
            self.valid_sample = lambda x: True

    @property
    def output_shape(self):
        return self.output_size

    def __len__(self):
        """
        Return the length of the iterator

        If it not was iterated once, it will compute the length by iterating
        from the entire dataset
        """
        if self.length is None:
            self.length = sum(1 for _ in self)
            self.__iter__()
        return self.length

    def __iter__(self):
        self.i = 0
        self.shuffler.start(self)
        return self

    def __next__(self):
        life, timestamp = self.shuffler.next_element(self.valid_sample)
        X, y, metadata = self.dataset[life]
        window = self.slicing_function(
            X, y, timestamp, self.window_size, self.output_size, self.add_last
        )
        return window[0], window[1], [self.sample_weight(y, timestamp, metadata)]

    def get_data(self):
        N_points = len(self)
        dimension = self.window_size * self.n_features
        X = np.zeros((N_points, dimension), dtype=np.float32)
        y = np.zeros((N_points, self.output_size), dtype=np.float32)
        sample_weight = np.zeros(N_points, dtype=np.float32)

        for i, (X_, y_, sample_weight_) in enumerate(self):
            X[i, :] = X_.flatten()
            y[i, :] = y_.flatten()
            sample_weight[i] = sample_weight_[0]
        return X, y, sample_weight

    @property
    def n_features(self) -> int:
        """Number of features of the transformed dataset
        This is a helper method to obtain the transformed
        dataset information from the WindowedDatasetIterator
        Returns
        -------
        int
            Number of features of the transformed dataset
        """
        return self.dataset.transformer.n_features

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Tuple containing (window_size, n_features)

        Returns
        -------
        Tuple[int, int]
            Tuple containing (window_size, n_features)
        """
        return (self.window_size, self.n_features)
