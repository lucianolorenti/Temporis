from copy import deepcopy
from typing import Any, List, Optional, Union

import pandas as pd
from temporis.transformation.functional.graph_utils import (
    root_nodes, topological_sort_iterator)
from temporis.transformation.functional.pipeline import (TemporisPipeline,
                                                         make_pipeline)
from temporis.transformation.functional.transformerstep import TransformerStep


class Joiner(TransformerStep):
    def transform(self, X: List[pd.DataFrame]):
        X_default = X[0]
        X_q = pd.concat(X[1:])
        missing_indices = X_default.index.difference(X_q.index)
        X_q = pd.concat((X_q, X_default.loc[missing_indices, :])).sort_index()
        return X_q


class Filter(TransformerStep):
    def __init__(
        self,
        value: Any,
        column: str,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.value = value
        self.column = column

    def transform(self, X):
        if self.column == "__category_all__":
            return X.drop(columns=[self.column])
        else:
            return X[X[self.column] == self.value].drop(columns=[self.column])


class SplitByCategory(TransformerStep):
    def __init__(
        self,
        categorical_feature_name: str,
        pipeline: TemporisPipeline,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.orig_pipeline = deepcopy(pipeline)
        self.categorical_feature_name = categorical_feature_name
        self._sub_pipelines = {}
        self.joiner = Joiner()(self)

    def _build_pipeline(self, category):
        self.disconnect(self.joiner)
        s = Filter(category, self.categorical_feature_name)(self)
        new_pipe = deepcopy(self.orig_pipeline)
        for node in topological_sort_iterator(new_pipe):
            node._name = "Category: {category} " + node.name
        for r in root_nodes(new_pipe):
            r(s)
        self.joiner(new_pipe.final_step)
        return new_pipe

    def transform(self, X):
        return X

    def partial_fit(self, X, y=None):

        if "default" not in self._sub_pipelines:
            self._sub_pipelines["default"] = self._build_pipeline("__category_all__")
        self.categorical_feature_name = self.find_feature(
            X, self.categorical_feature_name
        )
        for c in X[self.categorical_feature_name].unique():
            if c not in self._sub_pipelines:
                pipe = self._build_pipeline(c)
                self._sub_pipelines[c] = pipe
        return self

    def __call__(
        self, prev: Union[List["TransformerStepMixin"], "TransformerStepMixin"]
    ):
        self._add_previous(prev)
        return self.joiner
