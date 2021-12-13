from typing import List, Optional, Union

import numpy as np
import pandas as pd
from temporis.transformation import TransformerStep
from sklearn.pipeline import FeatureUnion, _transform_one

from temporis.transformation.features.tdigest import TDigest


class PandasToNumpy(TransformerStep):
    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values


class TransformerLambda(TransformerStep):
    def __init__(self, f, name: Optional[str] = None):
        super().__init__(name)
        self.f = f

    def transform(self, X, y=None):
        return self.f(X)


class IdentityTransformerStep(TransformerStep):
    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        if isinstance(input_array, pd.DataFrame):
            return input_array.copy()
        else:
            return input_array * 1


class SKLearnTransformerWrapper(TransformerStep):
    def __init__(self, transformer, name: Optional[str] = None):
        super().__init__(name)
        self.transformer = transformer

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        self.transformer.fit(X.values)
        return self

    def partial_fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        if hasattr(self.transformer, "partial_fit"):
            self.transformer.partial_fit(X.values)
        else:
            self.transformer.fit(X.values)
        return self

    def _column_names(self, X) -> List[str]:
        return X.columns

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input array must be a data frame")
        return pd.DataFrame(
            self.transformer.transform(X), columns=self._column_names(X), index=X.index
        )


def column_names_window(columns: list, window: int) -> list:
    """

    Parameters
    ----------
    columns: list
             List of column names

    window: int
            Window size

    Return
    ------
    Column names with the format: w_{step}_{feature_name}
    """
    new_columns = []
    for w in range(1, window + 1):
        for c in columns:
            new_columns.append(f"w_{w}_{c}")
    return new_columns


class QuantileEstimator:
    def __init__(self):
        self.tdigest_dict = None

    def update(self, X: pd.DataFrame):
        if X.shape[0] < 2:
            return self
        X = X.dropna()
        if self.tdigest_dict is None:
            self.tdigest_dict = {c: TDigest(100) for c in X.columns}

        for c in X.columns:
            self.tdigest_dict[c] = self.tdigest_dict[c].merge_unsorted(X[c].values)
        return self

    def estimate_quantile(
        self, q: float, feature: Optional[str] = None
    ) -> Union[pd.Series, float]:
        if feature is not None:
            return self.tdigest_dict[feature].estimate_quantile(q)
        else:
            return pd.Series(
                {
                    c: self.tdigest_dict[c].estimate_quantile(q)
                    for c in self.tdigest_dict.keys()
                }
            )
