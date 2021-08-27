from typing import List
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from numpy.lib.arraysetops import isin
from sklearn.pipeline import FeatureUnion, _transform_one

from temporis.transformation.functional.mixin import TransformerStepMixin


class Concatenate(TransformerStepMixin):


    def merge_dataframes_by_column(self, Xs):
        # indices = [X.index for X in Xs]
        # TODO: Check equal indices
        names = [n.name for n in self.previous]
        X = Xs[0]
        X.columns = [f'{names[0]}_{c}' for c in X.columns]
        for name, otherX in zip(names[1:], Xs[1:]):
            for c in otherX.columns:
                X[f'{name}_{c}'] = otherX[c]
        return X

    def transform(self, Xs:List[pd.DataFrame]):
        if not Xs:
            # All transformers are None
            return np.zeros((Xs[0].shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        return params
