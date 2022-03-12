from typing import List
from pyparsing import col
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from uncertainties import ufloat

from temporis.dataset.transformed import TransformedDataset
from tqdm.auto import tqdm
from collections import defaultdict
import antropy as ant

def entropy(s:np.ndarray):
    return ant.app_entropy(s)

def correlation(s: np.ndarray):
    N = s.shape[0]
    corr = spearmanr(s, np.arange(N), nan_policy="omit")
    if np.isnan(corr.correlation):
        corr = -100
    else:
        corr = corr.correlation
    return corr


def autocorrelation(s: np.ndarray):
    diff = np.diff(s)
    return np.sum(diff ** 2) / s.shape[0]


def monotonicity(s: np.ndarray):
    N = s.shape[0]
    diff = np.diff(s)
    return 1 / (N - 1) * np.abs(np.sum(diff > 0) - np.sum(diff < 0))


def n_unique(s: np.ndarray):
    return len(np.unique(s))

def null(s: np.ndarray):
    return np.isfinite(s)

def analysis(
    transformed_dataset: TransformedDataset,
    *,
    show_progress: bool,
    what_to_compute: List[str] = [],
):
    metrics = {
        "std": np.std,
        "correlation": correlation,
        "autocorrelation": autocorrelation,
        "monotonicity": monotonicity,
        'number_of_unique_elements': n_unique,
        'null': null,
        'entropy': entropy
    }
    if len(what_to_compute) == 0:
        what_to_compute = list(sorted(metrics.keys()))
    data = defaultdict(lambda: defaultdict(list))
    iterator = transformed_dataset
    if show_progress:
        iterator = tqdm(iterator)
    for (X, _, _) in iterator:
        for column_index in range(X.shape[1]):
            column_name = transformed_dataset.transformer.column_names[column_index]
            for what in what_to_compute:
                data[column_name][what].append(metrics[what](X[:, column_index]))

    data_df = defaultdict(lambda: defaultdict(list))
    for column_name in data.keys():
        for what in data[column_name]:
            data_df[column_name][f"{what} Mean"] = ufloat(
                np.mean(data[column_name][what]),
                np.std(data[column_name][what]),
            )
            data_df[column_name][f"{what} Max"] = np.max(
                np.mean(data[column_name][what])
            )
            data_df[column_name][f"{what} Min"] = np.min(
                np.mean(data[column_name][what])
            )
    return pd.DataFrame(data_df).T


def feature_analysis(transformed_dataset: TransformedDataset, show_progress: bool):
    """

    Bibliography:
    Remaining Useful Life Prediction of Machining Tools by 1D-CNN LSTM Network
    """
    data = {}
    iterator = transformed_dataset
    if show_progress:
        iterator = tqdm(iterator)

    for (X, y, sw) in iterator:
        N = X.shape[0]
        for j in range(X.shape[1]):
            corr = spearmanr(X[:, j], np.arange(N), nan_policy="omit")
            if np.isnan(corr.correlation):
                corr = -100
            else:
                corr = corr.correlation
            diff = np.diff(X[:, j])
            autocorrelation = np.sum(diff ** 2) / N
            monotonicity = 1 / (N - 1) * np.abs(np.sum(diff > 0) - np.sum(diff < 0))
            column_name = transformed_dataset.transformer.column_names[j]
            if column_name not in data:
                data[column_name] = {
                    "correlation": [],
                    "autocorrelation": [],
                    "monotonicity": [],
                }

            data[column_name]["correlation"].append(corr)
            data[column_name]["autocorrelation"].append(autocorrelation)
            data[column_name]["monotonicity"].append(monotonicity)
    df_data = {}
    for column_name in data.keys():

        df_data[column_name] = {
            "Correlation": ufloat(
                np.mean(data[column_name]["correlation"]),
                np.std(data[column_name]["correlation"]),
            ),
            "Autocorrelation": ufloat(
                np.mean(data[column_name]["autocorrelation"]),
                np.std(data[column_name]["autocorrelation"]),
            ),
            "Monotonicity": ufloat(
                np.mean(data[column_name]["monotonicity"]),
                np.std(data[column_name]["monotonicity"]),
            ),
        }
    return pd.DataFrame(df_data).T
