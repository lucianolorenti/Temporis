import shelve
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

from sklearn.base import TransformerMixin
from temporis import CACHE_PATH
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.transformation.functional.graph_utils import (
    dfs_iterator,
    root_nodes,
    topological_sort_iterator,
)
from temporis.transformation.functional.transformerstep import TransformerStep
from tqdm.auto import tqdm


def encode_tuple(tup: Tuple):
    return ",".join(list(map(lambda x: str(hash(x)), tup)))


def decode_tuple(s: str):
    return s.split(",")


class GraphTraversalCache:
    def __init__(self, root_nodes, dataset, cache_path: Path = CACHE_PATH):
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
        root_nodes :

        dataset :

        """
        filename = "".join(str(uuid.uuid4()).split("-"))
        self.cache_path = cache_path / "GraphTraversalCache" / filename
        self.cache_path.parent.mkdir(exist_ok=True, parents=True)

        self.transformed_cache = shelve.open(str(self.cache_path))
        for r in root_nodes:
            for i, df in enumerate(dataset):
                self.transformed_cache[encode_tuple((r, None, i))] = df

    def clear_cache(self):
        self.transformed_cache.close()
        self.transformed_cache = shelve.open(str(self.cache_path))

    def state_up_to(self, current_node: TransformerStep, dataset_element: int):

        previous_node = current_node.previous

        if len(previous_node) > 1:
            return [
                self.transformed_cache[encode_tuple((current_node, p, dataset_element))]
                for p in previous_node
            ]
        else:
            if len(previous_node) == 1:
                previous_node = previous_node[0]
            else:
                previous_node = None
            return self.transformed_cache[
                encode_tuple((current_node, previous_node, dataset_element))
            ]

    def clean_state_up_to(self, current_node: TransformerStep, dataset_element: int):

        previous_node = current_node.previous

        for p in previous_node:
            self.transformed_cache[
                encode_tuple((current_node, p, dataset_element))
            ] = None

    def store(
        self,
        next_node: Optional[TransformerStep],
        node: TransformerStep,
        dataset_element: int,
        new_element: pd.DataFrame,
    ):
        self.transformed_cache[
            encode_tuple((next_node, node, dataset_element))
        ] = new_element

    def remove_state(self, nodes: Union[TransformerStep, List[TransformerStep]]):
        if not isinstance(nodes, list):
            nodes = [nodes]
        for n in nodes:
            keys_to_remove = self.get_keys_of(n)
            for k in keys_to_remove:
                self.transformed_cache.pop(k)

    def advance_state_to(self, node):
        prev_key = self.previous_state_key(node)
        assert prev_key == node.previous
        self.transformed_cache[node.next][node] = self.transformed_cache.pop(node)[
            node.previous
        ]

    def get_keys_of(self, n):
        return [
            k
            for k in self.transformed_cache.keys()
            if decode_tuple(k)[0] == str(hash(n))
        ]


class CachedPipelineRunner:
    """Performs an execution of the transformation graph caching the intermediate results



    Parameters
    ----------
    final_step : TransformerStep
        Last step of the graph
    """

    def __init__(self, final_step: TransformerStep):

        self.final_step = final_step
        self.root_nodes = root_nodes(final_step)

    def _run(
        self,
        dataset: Iterable[pd.DataFrame],
        fit: bool = True,
        show_progress: bool = False,
    ):
        dataset_size = len(dataset)
        cache = GraphTraversalCache(self.root_nodes, dataset)

        for node in topological_sort_iterator(self.final_step):

            if isinstance(node, TransformerStep) and fit:
                for dataset_element in range(dataset_size):
                    node.partial_fit(cache.state_up_to(node, dataset_element))
            if show_progress:
                bar = tqdm(range(dataset_size))
            else:
                bar = range(dataset_size)
            for dataset_element in bar:
                if show_progress:
                    bar.set_description(node.name)
                new_element = node.transform(cache.state_up_to(node, dataset_element))

                cache.clean_state_up_to(node, dataset_element)
                cache.store(node.next, node, dataset_element, new_element)
            cache.remove_state(node)

        last_state_key = cache.get_keys_of(None)[0]
        return cache.transformed_cache[last_state_key]

    def fit(self, dataset: Iterable[pd.DataFrame], show_progress: bool = False):
        return self._run(dataset, fit=True, show_progress=show_progress)

    def transform(self, df: pd.DataFrame):
        return self._run([df], fit=False)


class NonCachedPipelineRunner:
    """Performs an execution of the transformation graph recomputing on each path



    Parameters
    ----------
    final_step : TransformerStep
        Last step of the graph
    """

    def __init__(self, final_step: TransformerStep):
        self.final_step = final_step
        self.root_nodes = root_nodes(final_step)

    def _run(self, dataset: Iterable[pd.DataFrame], fit: bool = True):
        raise NotImplementedError

    def fit(self, dataset: Iterable[pd.DataFrame], show_progress: bool = False):
        return self._run(dataset, fit=True, show_progress=show_progress)

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

    def fit(
        self,
        dataset: Union[AbstractTimeSeriesDataset, pd.DataFrame],
        show_progress: bool = False,
    ):
        if isinstance(dataset, pd.DataFrame):
            dataset = [dataset]
        c = self.runner.fit(dataset, show_progress=show_progress)
        self.column_names = c.columns
        self.fitted_ = True

        return self

    def transform(self, df: pd.DataFrame):
        return self.runner.transform(df)
