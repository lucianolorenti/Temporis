from typing import Optional, Union

import pandas as pd
from temporis.transformation import TransformerStep
import numpy as np
from temporis.transformation.features.tdigest import TDigest


class RobustMinMaxScaler(TransformerStep):
    """Scale features using statistics that are robust to outliers.

    This Scaler scales the data according to the quantile range
    The IQR is the range between the limits provided, by default, 
    1st quartile (25th quantile) and the 3rd quartile (75th quantile).

    The quantiles are approximated using tdigest

    Parameters
    ----------
    range : tuple
        Desired range of transformed data.
    clip : bool, optional
        Set to True to clip transformed values of held-out data to provided, by default True
    lower_quantile : float, optional
        Lower limit of the quantile range to compute the scale, by default 0.25
    upper_quantile : float, optional
        Upper limit of the quantile range to compute the scale, by default 0.75
    tdigest_size : Optional[int], optional
        Size of the t-digest structure, by default 100
    name : Optional[str], optional
        Name of the step, by default None
    """
    def __init__(
        self,
        range: tuple,
        clip: bool = True,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75,
        tdigest_size: Optional[int] = 100,
        name: Optional[str] = None,
    ):

        super().__init__(name)
        self.range = range
        self.min = range[0]
        self.max = range[1]
        self.data_min = None
        self.data_max = None
        self.clip = clip
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.tdigest_size = tdigest_size
        self.quantile_estimator = {

        }
        self.computed_limits = {

        }
        
    def partial_fit(self, df:pd.DataFrame, y=None):      
        if len(self.quantile_estimator) == 0:
            for c in df.columns:
                self.quantile_estimator[c] = TDigest(self.tdigest_size)
                self.computed_limits[c] = {'min':0, 'max': 0}

        for c in df.columns:
            self.quantile_estimator[c] = self.quantile_estimator[c].merge_unsorted(df[c].values)
            self.quantile_estimator[c] = self.quantile_estimator[c].merge_unsorted(df[c].values)

            self.computed_limits[c]['min'] = self.quantile_estimator[c].estimate_quantile(self.lower_quantile)
            self.computed_limits[c]['max'] = self.quantile_estimator[c].estimate_quantile(self.upper_quantile)

    
    def transform(self, X:pd.DataFrame):
        for c in X.columns:
            lower_limit = self.computed_limits[c]['min']
            upper_limit = self.computed_limits[c]['max']

            if self.clip:
                X[c].clip(lower=lower_limit, upper=upper_limit, inplace=True)

            
        
            t_data_c_std = (X[c] - lower_limit) / (upper_limit - lower_limit)
            
            X[c] = t_data_c_std * (self.max - self.min) + self.min
            

        return X
            
class MinMaxScaler(TransformerStep):
    """Transform features by scaling each feature to a given range.

    This transformer scales and translates each feature individually 
    such that it is in the given range on the training set.

    Parameters
    ----------
    range : tuple
        Desired range of transformed data.
    clip : bool, optional
        Set to True to clip transformed values of held-out data to provided, by default True
    name : Optional[str], optional
         Name of the step, by default None
    """
    def __init__(
        self,
        range: tuple,
        clip: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.range = range
        self.min = range[0]
        self.max = range[1]
        self.data_min = None
        self.data_max = None
        self.clip = clip

    def partial_fit(self, df, y=None):


        partial_data_min = df.min(skipna=True)
        partial_data_max = df.max(skipna=True)
        if self.data_min is None:
            self.data_min = partial_data_min
            self.data_max = partial_data_max
        else:
            self.data_min = pd.concat([self.data_min, partial_data_min], axis=1).min(
                axis=1
            )
            self.data_max = pd.concat([self.data_max, partial_data_max], axis=1).max(
                axis=1
            )
        return self

    def fit(self, df, y=None):
        self.data_min = df.min()
        self.data_max = df.max()
        return self

    def transform(self, X):
        try:
            X = (
                (X - self.data_min)
                / (self.data_max - self.data_min)
                * (self.max - self.min)
            ) + self.min
        except:
            print(X)
            print(self.data_min)
            print(self.data_max)
            raise
        if self.clip:
            X.clip(lower=self.min, upper=self.max, inplace=True)
        return X


class StandardScaler(TransformerStep):
    """Standardize features by removing the mean and scaling to unit variance.
    
    Parameters
    ----------
    name : Optional[str], optional
        Name of the step, by default None
        
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.std = None
        self.mean = None

    def partial_fit(self, df, y=None):
        partial_data_mean = df.mean()
        partial_data_std = df.std()
        if self.mean is None:
            self.mean = partial_data_mean
            self.std = partial_data_std
        else:
            self.mean = pd.concat([self.mean, partial_data_mean], axis=1).mean(axis=1)
            self.std = pd.concat([self.std, partial_data_std], axis=1).mean(axis=1)
        return self

    def fit(self, df, y=None):
        self.mean = df.mean()
        self.std = df.std()
        return self

    def transform(self, X):
        return (X - self.mean) / (self.std)





class ScaleInvRUL(TransformerStep):
    """
    Scale binary columns according the inverse of the RUL.
    Usually this will be used before a CumSum transformation

    Parameters
    ----------
    rul_column: str
                Column with the RUL
    """

    def __init__(self, rul_column: str, name: Optional[str] = None):
        super().__init__(name)
        self.RUL_list_per_column = {}
        self.penalty = {}
        self.rul_column_in = rul_column
        self.rul_column = None

    def partial_fit(self, X: pd.DataFrame):
        if self.rul_column is None:
            self.rul_column = self.column_name(X, self.rul_column_in)
        columns = [c for c in X.columns if c != self.rul_column]
        for c in columns:
            mask = X[X[c] > 0].index
            if len(mask) > 0:
                RUL_list = self.RUL_list_per_column.setdefault(c, [])
                RUL_list.extend(
                    (
                        1
                        + (
                            X[self.rul_column].loc[mask].values
                            / X[self.rul_column].max()
                        )
                    ).tolist()
                )

        for k in self.RUL_list_per_column.keys():

            self.penalty[k] = 1 / np.median(self.RUL_list_per_column[k])

    def transform(self, X: pd.DataFrame):
        columns = [c for c in X.columns if c != self.rul_column]
        X_new = pd.DataFrame(index=X.index)
        for c in columns:
            if c in self.penalty:
                X_new[c] = X[c] * self.penalty[c]
        return X_new


class PerCategoricalMinMaxScaler(TransformerStep):
    """Performs a minmax scaler partition the data trough some categorical feature

    Usually, different execution configurations lead to different scales in the features. 
    Therefore, sometimes it is useful to scale the data based on a categorical feature,   
    to reflect the difference in the execution parameters.

    Parameters
    ----------
    categorical_feature: str
        The name of the categorical feature whose values are going to be used
        to split each time-series
    scaler: Optional[Union[MinMaxScaler,RobustMinMaxScaler]], default MinMaxScaler
        The scale to use when scaling the data
    scaler_params: dict
        Parameters used when constructing the scaler
    name: Optional[str]
        Name of the transformer

    """  

    def __init__(
        self,
        categorical_feature: str,
        scaler: Optional[Union[MinMaxScaler,RobustMinMaxScaler]] = MinMaxScaler,
        scaler_params: dict = {},        
        name: Optional[str] = None,
    ):     
        super().__init__(name)                
        self.categorical_feature = categorical_feature
        self.categorical_feature_name = None
        self.scaler = scaler
        self.scaler_params = scaler_params
    
        self.scalers = {
            "default": self.scaler(**self.scaler_params)
            
        }

    def partial_fit(self, X, y=None):
        if self.categorical_feature_name is None:
            self.categorical_feature_name = self.find_feature(X, self.categorical_feature)
        for category, data in X.groupby(self.categorical_feature_name):
            data = data.drop(columns=[self.categorical_feature_name])
            if category not in self.scalers:
                self.scalers[category] = self.scaler(**self.scaler_params)               
            self.scalers[category].partial_fit(data)
            self.scalers["default"].partial_fit(data)

    def transform(self, X:pd.DataFrame):
     
        X_new = X.drop(columns=[self.categorical_feature_name])
  
        for category, data in X.groupby(self.categorical_feature_name):
            
            data = data.drop(columns=[self.categorical_feature_name])
            scaler = self.scalers[category] if category in self.scalers else self.scalers['default'] #Use a defaultdict
            X_new.loc[data.index, :] = scaler.transform(data)
        return X_new