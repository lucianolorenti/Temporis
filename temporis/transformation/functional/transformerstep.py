from copy import copy
from typing import List, Optional

import pandas as pd
from sklearn.base import TransformerMixin
from temporis.transformation.functional.mixin import TransformerStepMixin
from copy import copy


class TransformerStep(TransformerStepMixin,  TransformerMixin):

    def partial_fit(self, X, y=None):
        return self

    def fit(self, X, y=None):
        return self

    def column_name(self, df: pd.DataFrame, cname: str):
        columns = [c for c in df.columns if cname in c]
        if len(columns) == 0:
            raise ValueError("{cname} is not present in the dataset")
        return columns[0]



