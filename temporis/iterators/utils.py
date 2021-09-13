
from temporis.transformation.functional.transformers import TransformerIdentity
from numpy.lib.arraysetops import isin
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from typing import Union

import numpy as np
from temporis.iterators.batcher import Batcher
from temporis.iterators.iterators import WindowedDatasetIterator


def true_values(
    dataset_iterator: Union[WindowedDatasetIterator, Batcher, AbstractTimeSeriesDataset]
) -> np.array:
    """Obtain the true RUL of the dataset after the transformation

    Parameters
    ----------
    dataset_iterator : Union[WindowedDatasetIterator, Batcher, AbstractTimeSeriesDataset]
        Iterator of the dataset

    Returns
    -------
    np.array
         target values after the transformation
    """
    if isinstance(dataset_iterator, Batcher):
        dataset_iterator = dataset_iterator.iterator
    if isinstance(dataset_iterator, AbstractTimeSeriesDataset):
        t = TransformerIdentity()
        t.fit(dataset_iterator)

        dataset_iterator = WindowedDatasetIterator(
            dataset_iterator, window_size=1, transformer=t
        )
    d = np.concatenate([y for _, y, _ in dataset_iterator])
    return d
