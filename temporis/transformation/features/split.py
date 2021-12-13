from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
from temporis.transformation.functional.graph_utils import (
    root_nodes,
    topological_sort_iterator,
)
from temporis.transformation.functional.pipeline.pipeline import TemporisPipeline
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
        values: List[Any],
        columns: Union[List[str], str],
        name: Optional[str] = None,
    ):
        def prepare_value(v):
            if isinstance(v, str):
                return f"'{v}'"
            else:
                return v

        super().__init__(name)
        self.values = values
        self.columns = columns
        self.query = " & ".join(
            [f"({c} == {prepare_value(v)})" for c, v in zip(self.columns, self.values)]
        )

    def transform(self, X):
        if self.values == ["__category_all__"]:
            return X.drop(columns=self.columns)
        else:
            return X.query(self.query).drop(columns=self.columns)


class SplitByCategory(TransformerStep):
    def __init__(
        self,
        categorical_feature_names: Union[str, List[str]],
        pipeline: TemporisPipeline,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        if isinstance(categorical_feature_names, str):
            categorical_feature_names = [categorical_feature_names]

        self.orig_pipeline = deepcopy(pipeline)
        self.categorical_feature_names = categorical_feature_names

        self._categorical_feature_names_resolved = None
        self._sub_pipelines = {}
        self.joiner = Joiner()(self)

    def _build_pipeline(self, categories: Tuple[str]):
        self.disconnect(self.joiner)
        s = Filter(categories, self._categorical_feature_names_resolved)(self)
        new_pipe = deepcopy(self.orig_pipeline)
        for node in topological_sort_iterator(new_pipe):
            node.name = f"Category: {categories} " + node.name
        for r in root_nodes(new_pipe):
            r(s)
        self.joiner(new_pipe.final_step)
        return new_pipe

    def transform(self, X):
        return X

    def partial_fit(self, X, y=None):
        if self._categorical_feature_names_resolved is None:
            self._categorical_feature_names_resolved = [
                self.find_feature(X, f) for f in self.categorical_feature_names
            ]
        if "default" not in self._sub_pipelines:
            self._sub_pipelines["default"] = self._build_pipeline(["__category_all__"])
        for _, c in (
            X[self._categorical_feature_names_resolved].drop_duplicates().iterrows()
        ):
            c = tuple(c)
            if c not in self._sub_pipelines:
                self._sub_pipelines[c] = self._build_pipeline(c)
        return self

    def __call__(
        self, prev: Union[List["TransformerStepMixin"], "TransformerStepMixin"]
    ):
        self._add_previous(prev)
        return self.joiner
