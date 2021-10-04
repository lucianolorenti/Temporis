from typing import List
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
import numpy as np
import logging 

logger = logging.getLogger(__name__)

def histogram_per_life(
    ds: AbstractTimeSeriesDataset,
    feature: str,
    number_of_bins: int = 15,
    share_bins: bool = False,
    normalize: bool = True,
) -> List[np.ndarray]:

    shared_bins = None
    histograms = []
    for k, life in enumerate(ds):
        try:
            d = life[feature]
            if share_bins:
                if shared_bins is None:
                    bins_to_use = number_of_bins
            else:
                bins_to_use = number_of_bins
            h, b = np.histogram(d, bins=bins_to_use)
            if share_bins and shared_bins is None:
                shared_bins = b
            if normalize:
                h = h / np.sum(h)
            data = np.vstack(((b[0:-1] + b[1:])/2, h))
            histograms.append(data)
        except ValueError as e:
            logger.info(
                "Error {e} when computing the distribution for feature {c} in life {k}"
            )
    return histograms
