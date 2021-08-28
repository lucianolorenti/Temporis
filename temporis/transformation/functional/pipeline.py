from temporis.transformation.functional.graph_utils import (
    dfs_iterator,
    root_nodes,
    topological_sort_iterator,
)
from pandas.core.frame import DataFrame
from sklearn.base import TransformerMixin
from temporis.transformation.functional.concatenate import Concatenate
from temporis.transformation.functional.transformerstep import TransformerStep
from typing import Dict, Iterable, List, Optional, Union, final

from numpy.lib.arraysetops import isin


from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset

import pandas as pd
from copy import copy


class GraphTraversalCache:
    def __init__(self, root_nodes, dataset):
        """[summary]


        The cache data structures has the following form
        Current Node -> Previous Nodes -> [Transformed Dataset]

        * cache[n]:
            contains a dict with one key for each previous node
        * cache[n][n.previous[0]]
            A list with each element of the dataset transformed in
            up to n.previous[0]


        Parameters
        ----------
        root_nodes : [type]
            [description]
        dataset : [type]
            [description]
        """
        self.transformed_cache = {}
        for r in root_nodes:
            self.transformed_cache[r] = {None: {i: df for i, df in enumerate(dataset)}}

    def clear_cache(self):
        self.transformed_cache = {}

    def state_up_to(self, current_node: TransformerStep, dataset_element: int):

        previous_node = current_node.previous

        if len(previous_node) > 1:
            return [
                self.transformed_cache[current_node][p][dataset_element]
                for p in previous_node
            ]
        else:
            if len(previous_node) == 1:
                previous_node = previous_node[0]
            else:
                previous_node = None
            return self.transformed_cache[current_node][previous_node][dataset_element]

    def clean_state_up_to(self, current_node: TransformerStep, dataset_element: int):

        previous_node = current_node.previous

        for p in previous_node:
            self.transformed_cache[current_node][p][dataset_element] = None

    def store(
        self,
        next_node: Optional[TransformerStep],
        node: TransformerStep,
        dataset_element: int,
        new_element: pd.DataFrame,
    ):
        if next_node not in self.transformed_cache:
            self.transformed_cache[next_node] = {}

        if next_node is not None:
            previous_node = next_node.previous
        else:
            previous_node = [None]

        if node not in self.transformed_cache[next_node]:
            self.transformed_cache[next_node][node] = {}
        self.transformed_cache[next_node][node][dataset_element] = new_element

    def remove_state(self, nodes: Union[TransformerStep, List[TransformerStep]]):
        if not isinstance(nodes, list):
            nodes = [nodes]
        for n in nodes:
            self.transformed_cache.pop(n)

    def advance_state_to(self, node):
        if node.next not in self.transformed_cache:
            self.transformed_cache[node.next] = {}
        prev_key = self.previous_state_key(node)
        assert prev_key == node.previous
        self.transformed_cache[node.next][node] = self.transformed_cache.pop(node)[
            node.previous
        ]


class CachedPipelineRunner:
    """Performs an execution of the transformation graph caching the intermediate results



        Parameters
        ----------
        final_step : TransformerStep
            Last step of the graph
    """
    def __init__(self, final_step:TransformerStep):
        
        self.final_step = final_step
        self.root_nodes = root_nodes(final_step)

    def _run(self, dataset: Iterable[pd.DataFrame], fit: bool = True):
        dataset_size = len(dataset)
        cache = GraphTraversalCache(self.root_nodes, dataset)
        for node in topological_sort_iterator(self.final_step):
            if isinstance(node, TransformerStep) and fit:
                for dataset_element in range(dataset_size):
                    node.partial_fit(cache.state_up_to(node, dataset_element))
            for dataset_element in range(dataset_size):
                new_element = node.transform(cache.state_up_to(node, dataset_element))

                cache.clean_state_up_to(node, dataset_element)
                cache.store(node.next, node, dataset_element, new_element)
            cache.remove_state(node)

        last_state = cache.transformed_cache[None]
        last_graph_node = list(last_state.keys())[0]

        return last_state[last_graph_node][0]

    def fit(self, dataset: Iterable[pd.DataFrame]):
        return self._run(dataset, fit=True)

    def transform(self, df: pd.DataFrame):
        return self._run([df], fit=False)


class NonCachedPipelineRunner:
    """Performs an execution of the transformation graph recomputing on each path

        

        Parameters
        ----------
        final_step : TransformerStep
            Last step of the graph
    """
    def __init__(self, final_step:TransformerStep):        
        self.final_step = final_step
        self.root_nodes = root_nodes(final_step)

    def _run(self, dataset: Iterable[pd.DataFrame], fit: bool = True):        
        raise NotImplementedError

        

    def fit(self, dataset: Iterable[pd.DataFrame]):
        return self._run(dataset, fit=True)

    def transform(self, df: pd.DataFrame):
        return self._run([df], fit=False)


class TemporisPipeline(TransformerMixin):
    def __init__(self, final_step):
        self.final_step = final_step
        self.fitted_ = False
        self.runner = CachedPipelineRunner(final_step)

    def find_node(
        self, name: str
    ) -> Union[List[TransformerStep], TransformerStep, None]:
        matches = []
        for node in dfs_iterator(self.final_step):
            if node.name == name:
                matches.append(node)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            return matches
        else:
            return None

    def fit(self, dataset: Union[AbstractTimeSeriesDataset, pd.DataFrame]):
        if isinstance(dataset, pd.DataFrame):
            dataset = [dataset]
        c = self.runner.fit(dataset)
        self.column_names = c.columns
        self.fitted_ = True

        return self

    def transform(self, df: pd.DataFrame):
        return self.runner.transform(df)
