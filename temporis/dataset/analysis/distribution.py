from typing import List, Tuple
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
import numpy as np
import logging 
from itertools import combinations
from scipy.special import kl_div
import pandas as pd 

logger = logging.getLogger(__name__)

def histogram_per_life(
    ds: AbstractTimeSeriesDataset,
    feature: str,
    number_of_bins: int = 15,
    share_bins: bool = False,
    normalize: bool = True,
) -> List[np.ndarray]:

  
    if share_bins:
        min_value = ds[0][feature].min()
        max_value = ds[0][feature].max()
        for life in ds:
            min_value = min(np.min(life[feature]), min_value)
            max_value = max(np.max(life[feature]), max_value)
        bins_to_use = np.linspace(min_value, max_value, number_of_bins+1)
    else:
        bins_to_use = number_of_bins

    histograms = []
    for k, life in enumerate(ds):
        try:
            d = life[feature]
            h, b = np.histogram(d, bins=bins_to_use)
            
            if normalize:
                h = h / np.sum(h)
                h += 1e-50
            data = np.vstack(((b[0:-1] + b[1:])/2, h))
            histograms.append(data)
        except Exception as e:
            logger.info(
                f"Error {e} when computing the distribution for feature {feature} in life {k}"
            )
    return histograms


def divergence_of_features(
    ds: AbstractTimeSeriesDataset,
    feature: str,
    number_of_bins: int = 15,
) ->  List[float]:
    h = histogram_per_life(ds, feature, number_of_bins, share_bins=True, normalize=True)
    return [kl_div(h1[1, :], h2[1, :]) for h1, h2 in combinations(h, 2)]


def features_divergeces(ds:AbstractTimeSeriesDataset, number_of_bins:int=15) -> Tuple[pd.DataFrame, dict]: 
    data = {}
    divergences = []
    for feature in ds.numeric_features():
        feature_divergences = divergence_of_features(ds, feature, number_of_bins=number_of_bins)
        data[feature] = [np.mean(feature_divergences)]
        divergences.append((feature, np.nanmean(feature_divergences)))
      
    df =  pd.DataFrame(divergences, columns=['Feature', 'Mean divergence']).set_index('Feature')

    return df, divergences
        