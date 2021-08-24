from typing import Optional


from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from sklearn.pipeline import Pipeline




def transform_given_list(X, steps):
    for t in steps:
        if t == 'passthrough':
            continue
        X = t.transform(X)
    return X


class LivesPipeline(Pipeline):
    """Transformer pipeline 

    Parameters
    ----------
    Pipeline : [type]
        [description]
    """
    def fit(self, X, y=None, apply_before=None):
        """Fit the pipeline given a dataset


        The fitting procedure works by traversing the 
        graph by levels, fitting all of the lives in the level i,
        before fitting the level i+1 

        Parameters
        ----------
        X : Dataset
            Dataset in which fit the data
        y : [type], optional
            [description], by default None
        apply_before : [type], optional
            [description], by default None
        """

        if not isinstance(X, AbstractTimeSeriesDataset):
            super().fit(X, y)

        from temporis.transformation.featureunion import PandasFeatureUnion
        estimators = apply_before.copy() if apply_before is not None else []

        for _, est in self.steps:
            if est == 'passthrough':
                continue
            if isinstance(est, LivesPipeline):
                est.fit(X, apply_before=estimators)
            elif isinstance(est, PandasFeatureUnion):
                est.fit(X, apply_before=estimators)
            else:
                for life in X:
                    est.partial_fit(transform_given_list(life, estimators))
            estimators.append(est)
