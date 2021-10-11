from typing import List, Optional

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset


def variance_information(
    dataset: AbstractTimeSeriesDataset,
    features: Optional[List[str]] = None,
    transformer=None,
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
                  ['Feature', 'Max Std', 'Mean Std']

    dict: string -> list
          The key is the column name and the value is the list of std proportion
          for each life
    """
    if transformer is not None:
        common_features = []
        for life in dataset:
            life = transformer.transform(life)
            common_features.append(set(life.columns.tolist()))
        common_features = common_features[0].intersection(*common_features)
    else:
        common_features = dataset.common_features()
    if features:
        common_features = set(common_features).intersection(set(features))

    std_per_life = {}
    for life in dataset:
        if transformer is not None:
            life = transformer.transform(life)
        d = life.std().to_dict()
        for column in common_features:
            if column not in d:
                continue
            if not isinstance(d[column], float):
                continue
            std_list = std_per_life.setdefault(column, [])
            std_list.append(d[column])

    data = [
        (
            column,
            np.min(std_per_life[column]),
            np.mean(std_per_life[column]),
            np.max(std_per_life[column]),
        )
        for column in std_per_life.keys()
    ]

    df = pd.DataFrame(
        data,
        columns=[
            "Feature",
            "Min std",
            "Mean std",
            "Max std",
        ],
    )
    df.sort_values(by="Min std", inplace=True, ascending=True)
    return df, std_per_life
