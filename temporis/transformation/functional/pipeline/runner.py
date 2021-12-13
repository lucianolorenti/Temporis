
from typing import Iterable

import pandas as pd
from temporis.transformation.functional.graph_utils import (
    root_nodes, topological_sort_iterator)
from temporis.transformation.functional.pipeline.cache_store import CacheStoreType
from temporis.transformation.functional.pipeline.traversal import \
    CachedGraphTraversal
from temporis.transformation.functional.transformerstep import TransformerStep
from tqdm.auto import tqdm


class CachedPipelineRunner:
    """Performs an execution of the transformation graph caching the intermediate results



    Parameters
    ----------
    final_step : TransformerStep
        Last step of the graph
    """

    def __init__(self, final_step: TransformerStep, cache_type:CacheStoreType = CacheStoreType.SHELVE):

        self.final_step = final_step
        self.root_nodes = root_nodes(final_step)
        self.cache_type = cache_type

    def _run(
        self,
        dataset: Iterable[pd.DataFrame],
        fit: bool = True,
        show_progress: bool = False,
    ):
        dataset_size = len(dataset)

        with CachedGraphTraversal(self.root_nodes, dataset, cache_type=self.cache_type) as cache:
            for node in topological_sort_iterator(self.final_step):
                if isinstance(node, TransformerStep) and fit:
                    for dataset_element in range(dataset_size):
                        d = cache.state_up_to(node, dataset_element)
                        node.partial_fit(d)

                if show_progress:
                    bar = tqdm(range(dataset_size))
                else:
                    bar = range(dataset_size)
                for dataset_element in bar:
                    if show_progress:
                        bar.set_description(node.name)
                    new_element = node.transform(cache.state_up_to(node, dataset_element))
                    cache.clean_state_up_to(node, dataset_element)
                
                    if len(node.next) > 0:
                        for n in node.next:
                            cache.store(n, node, dataset_element, new_element)
                    else:
                        cache.store(None, node, dataset_element, new_element)
                cache.remove_state(node)
            last_state_key = cache.get_keys_of(None)[0]
            return cache.transformed_cache[last_state_key]

    def fit(self, dataset: Iterable[pd.DataFrame], show_progress: bool = False):
        return self._run(dataset, fit=True, show_progress=show_progress)

    def transform(self, df: pd.DataFrame):
        return self._run([df], fit=False)
