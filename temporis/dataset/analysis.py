import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
import antropy as ant
from itertools import combinations

logger = logging.getLogger(__name__)


def common_features_null_proportion_below(dataset: AbstractTimeSeriesDataset, t: float):
    """
    Return the list of features such that each feature in each
    life have a null proportion smaller than the threshold
    """

    def null_prop_all_below_threshold(c, t):
        return np.all(np.array(c) < t)

    _, null_proportion_per_life = null_proportion(dataset)

    return [
        column
        for column in null_proportion_per_life.keys()
        if null_prop_all_below_threshold(null_proportion_per_life[column], t)
    ]


def null_proportion(dataset: AbstractTimeSeriesDataset):
    """
    Return mean and max null proportion for each column of each life of the dataset

    Parameters
    ----------
    dataset: AbstractTimeSeriesDataset

    Return
    ------
    pd.DataFrame: Dataframe that contains three columns
                  ['Feature', 'Max Null Proportion', 'Mean Null Proportion']

    dict: string -> list
          The key is the column name and the value is the list of null proportion
          for each life
    """
    common_features = dataset.common_features()

    null_proportion_per_life = {}
    for life in dataset:
        d = life.isnull().mean().to_dict()
        for column in common_features:
            null_proportion_list = null_proportion_per_life.setdefault(column, [])
            null_proportion_list.append(d[column])

    for column in null_proportion_per_life.keys():
        null_proportion_per_life[column] = np.array(null_proportion_per_life[column])

    data = [
        (
            column,
            np.max(null_proportion_per_life[column]),
            np.mean(null_proportion_per_life[column]),
            np.sum(null_proportion_per_life[column] > 0.8),
        )
        for column in null_proportion_per_life.keys()
    ]

    df = pd.DataFrame(
        data,
        columns=[
            "Feature",
            "Max Null Proportion",
            "Mean Null Proportion",
            "Number of lives with more than 80% missing",
        ],
    )
    df.sort_values(by="Max Null Proportion", inplace=True, ascending=False)
    return df, null_proportion_per_life


def null_proportion_per_life(dataset: AbstractTimeSeriesDataset):
    """"""
    data = []
    for i, life in enumerate(dataset):
        null_prop = life.isnull().mean()
        number_of_completely_null = np.sum(null_prop > 0.99999)
        number_of_half_null = np.sum(null_prop > 0.5)
        number_of_25p_null = np.sum(null_prop > 0.25)
        mean_null_proportion = null_prop.mean().mean()
        data.append(
            (
                i,
                life.shape[1],
                number_of_completely_null,
                number_of_half_null,
                number_of_25p_null,
                mean_null_proportion,
            )
        )
    df = pd.DataFrame(
        data,
        columns=[
            "Life",
            "Number of features",
            "Number of completely null features",
            "N of features with 50% null",
            "N of features with 25% null",
            "Mean null propotion",
        ],
    )
    df.sort_values(
        by="Number of completely null features", inplace=True, ascending=False
    )
    return df


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


def entropy_information(
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
                  ['Feature', 'Max Null Proportion', 'Mean Null Proportion']

    dict: string -> list
          The key is the column name and the value is the list of null proportion
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

    entropy_per_life = {}
    for life in dataset:
        if transformer is not None:
            life = transformer.transform(life)
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


def correlation_analysis(
    dataset: AbstractTimeSeriesDataset,
    corr_threshold: float = 0,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Correlation Analysis
    Compute the correlation between all the features given an Iterable of executions.
    Parameters
    ---------
    dataset: AbstractTimeSeriesDataset
        Dataset of time series
    corr_threshold: float
        Treshold to consider two features of a single execution higly correlated
    features: Optional[List[str]], default None
        List of features to consider when computing the correlations
    Returns
    -------
    pd.DataFrame:
    * A DataFrame with three columns:
        * Feature name 1
        * Feature name 2
        * Percentage of time-series with a high correlation
        * Mean correlation across the time-series
        * Std correlation across the time-series
        * Mean Abs correlation across the time-series
        * Std Abs correlation across the time-series
        * Max correlation across the time-series
        * Min correlation across the time-series
    """
    if features is None:
        features = sorted(list(dataset.common_features()))        
    else:
        features = sorted(list(set(features).intersection(dataset.common_features())))
    features = dataset[0][features].corr().columns
    correlated_features = []
    for ex in dataset:
        ex = ex[features]
        corr_m = ex.corr().fillna(0)

        correlated_features_for_execution = []

        for f1, f2 in combinations(features, 2):
            if f1 == f2:
                continue

            correlated_features_for_execution.append((f1, f2, corr_m.loc[f1, f2]))

        correlated_features.extend(correlated_features_for_execution)

    df = pd.DataFrame(correlated_features, columns=["Feature 1", "Feature 2", "Corr"])
    output = df.groupby(by=["Feature 1", "Feature 2"]).mean()
    output.rename(columns={"Corr": "Mean Correlation"}, inplace=True)
    output["Std Correlation"] = df.groupby(by=["Feature 1", "Feature 2"]).std()

    def percentage_above_treshold(x):
        return (x["Corr"].abs() > corr_threshold).mean() * 100

    output["Percentage of lives with a high correlation"] = df.groupby(
        by=["Feature 1", "Feature 2"]
    ).apply(percentage_above_treshold)
    
    output["Abs mean correlation"] = df.groupby(by=["Feature 1", "Feature 2"]).apply(lambda x: x.abs().mean())
    output["Std mean correlation"] = df.groupby(by=["Feature 1", "Feature 2"]).apply(lambda x: x.abs().std())
    output["Max correlation"] = df.groupby(by=["Feature 1", "Feature 2"]).max()
    output["Min correlation"] = df.groupby(by=["Feature 1", "Feature 2"]).min()
    return output
