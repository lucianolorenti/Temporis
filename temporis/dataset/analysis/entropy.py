
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
import antropy as ant
from itertools import combinations

def entropy_information(
    dataset: AbstractTimeSeriesDataset,
    features: Optional[List[str]] = None
):
    """
    Return mean and max null proportion for each column of each life of the dataset

    Parameters
    ----------
    dataset: AbstractTimeSeriesDataset
             The dataset

    features: Optional[List[str]]=None
              Features to select

    transformer:
        Transformer

    Return
    ------
    pd.DataFrame: Dataframe that contains three columns
                  ['Feature', 'Max Null Proportion', 'Mean Null Proportion']

    dict: string -> list
          The key is the column name and the value is the list of null proportion
          for each life
    """
    common_features = dataset.common_features()
    if features:
        common_features = set(common_features).intersection(set(features))

    entropy_per_life = {}
    for life in dataset:
        d = {}
        for c in common_features:
            try:
                d[c] = ant.app_entropy(life[c].fillna(value=0).values)
            except TypeError:
                pass
        for column in common_features:
            if column not in d:
                continue
            if not isinstance(d[column], float):
                continue
            entropy_list = entropy_per_life.setdefault(column, [])
            entropy_list.append(d[column])

    data = [
        (
            column,
            np.min(entropy_per_life[column]),
            np.mean(entropy_per_life[column]),
            np.max(entropy_per_life[column]),
        )
        for column in entropy_per_life.keys()
    ]

    df = pd.DataFrame(
        data,
        columns=[
            "Feature",
            "Min entropy",
            "Mean entropy",
            "Max entropy",
        ],
    )
    df.sort_values(by="Min entropy", inplace=True, ascending=True)
    return df, entropy_per_life
