from typing import Union

import numpy as np
from temporis.iterators.batcher import Batcher
from temporis.iterators.iterators import WindowedDatasetIterator
from temporis.transformation.pipeline import LivesPipeline


def true_values(dataset_iterator: Union[WindowedDatasetIterator, Batcher]) -> np.array:
    """Obtain the true RUL of the dataset after the transformation

    Parameters
    ----------
    dataset_iterator : Union[WindowedDatasetIterator, Batcher]
        Iterator of the dataset

    Returns
    -------
    np.array
         target values after the transformation
    """
    if isinstance(dataset_iterator, Batcher):
        dataset_iterator = dataset_iterator.iterator 
    d =  np.concatenate([y for _, y, _ in dataset_iterator])
    return d
