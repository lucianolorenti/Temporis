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
        return ((1 / (y.iloc[i]+1)))

class InverseToLengthWeighted(AbstractSampleWeights):
    def __call__(self, y, i: int, metadata):
        return (1 / y[0])


SampleWeight = Union[AbstractSampleWeights,
                     Callable[[np.ndarray, int, Any], float]]


class WindowedDatasetIterator:
    def __init__(self,
                 dataset: TransformedDataset,
                 window_size: int,
                 step: int = 1,
                 output_size: int = 1,
                 shuffler: AbstractShuffler = NotShuffled(),
                 evenly_spaced_points: Optional[int] = None,
                 sample_weight:  SampleWeight= NotWeighted(),
                 add_last: bool = True,
                 discard_threshold: Optional[float] = None):

        self.dataset = dataset 
        self.shuffler = shuffler
        self.shuffler.initialize(self)
        self.evenly_spaced_points = evenly_spaced_points
        self.window_size = window_size
        self.step = step
        if not isinstance(sample_weight, AbstractSampleWeights) or not callable(sample_weight):
            raise ValueError('sample_weight should be an AbstractSampleWeights or a callable')

        self.sample_weight = sample_weight
        self.discard_threshold = discard_threshold
        
        self.i = 0
        self.output_size = output_size
        self.add_last = add_last


        self.length = None

    def _generate_input_sample_points_per_life(self, life_index:int):
        def window_evenly_spaced(y, i):
            w = y[i-self.window_size:i+1].diff().dropna().abs()
            return np.all(w <= self.evenly_spaced_points)


        if self.discard_threshold is not None:
            def should_discard(y, i):
                return y[i] > self.discard_threshold
        else:
            def should_discard(y, i): return False

        X, y, sw = self.dataset[life_index]
        self.processed_lives.append(life_index)
       

        list_ranges = range(self.window_size-1, y.shape[0], self.step)
        if self.evenly_spaced_points is not None:
            is_valid_point = window_evenly_spaced
        else:
            def is_valid_point(y, i): return True

        list_ranges = [
            i for i in list_ranges if is_valid_point(y, i) and not should_discard(y, i)
        ]

        for i in list_ranges:
            self.points.append((life_index, i, self._sample_weight(y, i, sw)))
    
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
        life, timestamp = self.shuffler.next_element()
        while timestamp < self.window_size -1:
            life, timestamp = self.shuffler.next_element()
        X, y, metadata = self.dataset[life]
        window = windowed_signal_generator(
            X, y, timestamp, self.window_size, self.output_size, self.add_last)
        return window[0], window[1], [self.sample_weight(y, timestamp, metadata)]

        for i in list_ranges:
            self.points.append((life_index, i, self._sample_weight(y, i, sw)))
            
    def _sample_weight(self, y, i: int, metadata):
        return self.sample_weight(y, i, metadata)
    
    def __getitem__(self, i: int):
        life, timestamp = self.shuffler.next_element(self)
        X, y, metadata = self.dataset[life]
        window = windowed_signal_generator(
            X, y, timestamp, self.window_size, self.output_size, self.add_last)
        return window[0], window[1], [self.sample_weights[i]]

    
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


def windowed_signal_generator(signal_X, signal_y, i: int, window_size: int, output_size: int = 1,  add_last: bool = True):
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
    initial = max(i - window_size+1, 0)
    signal_X_1 = signal_X[initial:i + (1 if add_last else 0), :]
    if len(signal_y.shape) == 1:

        signal_y_1 = signal_y[i:min(i+output_size, signal_y.shape[0])]

        if signal_y_1.shape[0] < output_size:
            padding = np.zeros(output_size - signal_y_1.shape[0])
            signal_y_1 = np.hstack((signal_y_1, padding))
        signal_y_1 = np.expand_dims(signal_y_1, axis=1)
    else:
        signal_y_1 = signal_y[i:min(i+output_size, signal_y.shape[0]), :]

        if signal_y_1.shape[0] < output_size:
            padding = np.zeros(
                ((output_size - signal_y_1.shape[0]), signal_y_1.shape[1]))
            signal_y_1 = np.concatenate((signal_y_1, padding), axis=0)

    if signal_X_1.shape[0] < window_size:

        signal_X_1 = np.vstack((
            np.zeros((
                window_size - signal_X_1.shape[0],
                signal_X_1.shape[1])),
            signal_X_1))

    return (signal_X_1, signal_y_1)


