from typing import Callable, Optional, Tuple

import numpy as np
import math


class AbstractShuffler:
    class Iterator:
        def __init__(self, shuffler, iterator):
            self.iterator = iterator
            self.shuffler = shuffler

        def __iter__(self):
            self.shuffler.initialize(self.iterator)
            return self

        def __next__(self):
            return self.shuffler.next_element()

    def iterator(self, iterator: "WindowedDatasetIterator"):
        return AbstractShuffler.Iterator(self, iterator)

    def at_end(self) -> bool:
        return self.current_time_series == self.wditerator.dataset.n_time_series

    def next_element(
        self, valid_timestmap: Optional[Callable[[int], bool]] = None
    ) -> Tuple[int, int]:

        valid = False
        while not valid:
            if self.at_end():
                raise StopIteration
            ts_index = self.time_series()
            timestamp = self.timestamp()
            if valid_timestmap:
                valid = valid_timestmap(timestamp)
            else:
                valid = True
        return ts_index, timestamp

    def start(self, iterator: "WindowedDatasetIterator"):
        self.initialize(iterator)

    def time_series(self) -> int:
        return self.current_time_series

    def initialize(self, iterator: "WindowedDatasetIterator"):
        self.wditerator = iterator
        self._samples_per_time_series = (
            np.ones(self.wditerator.dataset.n_time_series, dtype=np.int) * -1
        )
        self._time_series_sizes = (
            np.ones(self.wditerator.dataset.n_time_series, dtype=np.int) * -1
        )
        self.current_time_series = 0

    def load_time_series(self, time_series_index: int):
        N = self.wditerator.dataset.number_of_samples_of_time_series(time_series_index)
        self._samples_per_time_series[time_series_index] = math.ceil(
            N / self.wditerator.step
        )
        self._time_series_sizes[time_series_index] = N

    def number_samples_of_time_series(self, time_series_index: int) -> int:
        if self._samples_per_time_series[time_series_index] == -1:
            self.load_time_series(time_series_index)
        return self._samples_per_time_series[time_series_index]

    def number_of_samples_of_current_time_series(self):
        return self.number_samples_of_time_series(self.current_time_series)

    def timestamp(self):
        raise NotImplementedError

    def time_series_size(self, time_series_index: int):
        if self._time_series_sizes[time_series_index] == -1:
            self.load_time_series(time_series_index)
        return self._time_series_sizes[time_series_index]

    def current_time_series_size(self):
        return self.time_series_size(self.current_time_series)


class IntraTimeSeriesShuffler(AbstractShuffler):
    """
    Each point of the time series is shuffled, but the TS are kept in order

    Iteration 1: | TS 1 | TS 1 | Life 1 | Life 2 | Life 2 | Life 2
                    |   3    |  1     |  2     |   2    |   3    |   1
    Iteration 2: | TS 1 | TS 1 | TS 1 | TS 2 | TS 2 | TS 2
                    |   1    |  3     |  2     |   3    |   2    |   1
    """

    def time_series_changed(self):
        self.current_timestamp_index = 0
        if self.current_time_series == self.wditerator.dataset.n_time_series:
            return
        self.timestamps = np.arange(
            start=0,
            stop=self.current_time_series_size(),
            step=self.wditerator.step,
            dtype=np.uint,
        )
        np.random.shuffle(self.timestamps)

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.current_time_series = 0
        self.time_series_changed()
        self.n_time_series = iterator.dataset.n_time_series

    def timestamp(self) -> int:
        ret = self.timestamps[self.current_timestamp_index]
        self.current_timestamp_index += 1
        if (
            self.current_timestamp_index
            >= self.number_of_samples_of_current_time_series()
        ):
            self.current_time_series += 1
            self.time_series_changed()
        return ret


class TimeSeriesOrderShuffling(AbstractShuffler):
    """
    Time series are shuffled, but each point inside the time series kept  its order

    Iteration 1: | TS 1 | TS 1 | TS 1 | TS 2 | TS 2 | tS 2 |
                 |   1  | 2    |  3   |   1  |   2  |   3  |
    Iteration 2: | TS 2 | TS 2 | TS 2 | TS 1 | TS 1 | TS 1 |
                 |   1  |  2   |  3   |   1  |   2  |   3  |
    """

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.available_time_series = np.arange(
            iterator.dataset.n_time_series, dtype=np.uint
        )
        np.random.shuffle(self.available_time_series)
        self.available_time_series_index = 0
        self.current_timestamp = 0
        self.current_time_series = self.available_time_series[
            self.available_time_series_index
        ]

    def timestamp(self) -> int:
        ret = self.current_timestamp
        self.current_timestamp += self.wditerator.step
        if self.current_timestamp >= self.current_time_series_size():
            self.time_series_changed()
        return ret

    def time_series_changed(self):
        self.current_timestamp = 0
        self.available_time_series_index += 1
        if self.available_time_series_index < len(self.available_time_series):
            self.current_time_series = self.available_time_series[
                self.available_time_series_index
            ]
        else:
            self.current_time_series = len(self.available_time_series)


class TimeSeriesOrderIntraSignalShuffling(AbstractShuffler):
    """
     Each point in the ts is shuffled, and the ts order are shuffled also.

    Iteration 1: | TS 1 | TS 1 | TS 1 | TS 2 | TS 2 | TS 2
                 |   3  | 2    |  1   |   1  |   3  |   2
    Iteration 2: | TS 2 | TS 2 | TS 2 | TS 1 | TS 1 | TS 1
                  |   3 |  1   |  2   |   3  |   1  |   2

    """

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.available_time_series = np.arange(
            iterator.dataset.n_time_series, dtype=np.uint
        )
        np.random.shuffle(self.available_time_series)
        self.available_time_series_index = 0
        self.current_time_series = self.available_time_series[
            self.available_time_series_index
        ]

        self.available_time_stamps = np.arange(
            start=0,
            stop=self.current_time_series_size(),
            step=self.wditerator.step,
            dtype=np.uint,
        )
        self.available_time_stamps_index = 0

    def timestamp(self) -> int:
        ret = self.available_time_stamps[self.available_time_stamps_index]
        self.available_time_stamps_index += 1
        if (
            self.available_time_stamps_index
            == self.number_of_samples_of_current_time_series()
        ):
            self.time_series_changed()
        return ret

    def time_series_changed(self):
        self.available_time_stamps_index = 0
        self.available_time_series_index += 1
        if self.available_time_series_index < len(self.available_time_series):
            self.current_time_series = self.available_time_series[
                self.available_time_series_index
            ]
            self.available_time_stamps = np.arange(
                start=0,
                stop=self.current_time_series_size(),
                step=self.wditerator.step,
                dtype=np.uint,
            )
        else:
            self.current_time_series = len(self.available_time_series)


class InverseOrder(AbstractShuffler):
    """
    The data points will be fed in RUL decreasing order

    Iteration 1: | TS 2 | TS 1 | TS 2 | TS 1 | TS 2 | TS 1
                 |   4  | 3    |  3   |   2  |   2  |   1
    """

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.sizes = np.array(
            [self.time_series_size(i) for i in range(iterator.dataset.n_time_series)],
            dtype=np.int,
        )

    def time_series(self) -> int:
        self.current_time_series = np.argmax(self.sizes)
        return self.current_time_series

    def timestamp(self):
        ret = self.sizes[self.current_time_series] - 1
        self.sizes[self.current_time_series] = np.clip(
            np.int(self.sizes[self.current_time_series]) - self.wditerator.step,
            0,
            np.inf,
        )
        if np.sum(self.sizes) == 0:
            self.current_time_series = len(self.sizes)
        return ret


class AllShuffled(AbstractShuffler):
    """
    Everything is shuffled

    Iteration 1: | TS 1 | TS 2 | TS 2 | TS 1 | TS 1 | TS 2
                 |   3  | 2    |  1   |   1  |   2  |   3
    """

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.timestamps_per_ts = {
            i: None for i in range(self.wditerator.dataset.n_time_series)
        }
        self.timestamps_per_ts_indices = np.array(
            [0 for i in range(self.wditerator.dataset.n_time_series)], dtype=np.int
        )

    def time_series(self) -> int:
        l = np.random.randint(self.wditerator.dataset.n_time_series)

        while (
            self.number_samples_of_time_series(l) == self.timestamps_per_ts_indices[l]
        ):
            l = np.random.randint(self.wditerator.dataset.n_time_series)
        self.current_time_series = l

        return self.current_time_series

    def at_end(self) -> bool:

        return np.sum(self._samples_per_time_series) == np.sum(
            self.timestamps_per_ts_indices
        )

    def timestamp(self) -> int:
        ret = self.timestamps_per_ts[self.current_time_series][
            self.timestamps_per_ts_indices[self.current_time_series]
        ]
        self.timestamps_per_ts_indices[self.current_time_series] += 1
        return ret

    def load_time_series(self, time_series_index: int):
        super().load_time_series(time_series_index)
        self.timestamps_per_ts[time_series_index] = np.arange(
            start=0,
            stop=self.time_series_size(time_series_index),
            step=self.wditerator.step,
        )
        np.random.shuffle(self.timestamps_per_ts[time_series_index])


class NotShuffled(AbstractShuffler):
    """
    Not Shuffled
    Iteration 1: | Life 1 | Life 1  | Life 1 | Life 2 | Life 3 | Life 3
                  |   1    | 2      |  3     |   1    |   2    |   1
    """

    def initialize(self, iterator: "WindowedDatasetIterator"):
        super().initialize(iterator)
        self.current_time_series = 0
        self.current_timestamp = 0

    def time_series_changed(self):
        self.current_time_series += 1
        self.current_timestamp = 0

    def timestamp(self) -> int:
        ret = self.current_timestamp
        self.current_timestamp += self.wditerator.step
        if self.current_timestamp >= self.current_time_series_size():
            self.time_series_changed()
        return ret
