from pandas.core.frame import DataFrame
from sklearn.base import TransformerMixin
from temporis.transformation.functional.concatenate import Concatenate
from temporis.transformation.functional.transformerstep import TransformerStep
from typing import Dict, List, Optional, Union

from numpy.lib.arraysetops import isin


from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset

import pandas as pd
from copy import copy


def root_nodes(final_step):
    visited = set([final_step])
    to_process = copy(final_step.previous)

    while len(to_process) > 0:
        t = to_process.pop()
        if t not in visited:
            visited.add(t)
            to_process.extend(t.previous)

    return [n for n in visited if len(n.previous) == 0]


class TemporisPipeline(TransformerMixin):
    def __init__(self, final_step):
        self.root_nodes = root_nodes(final_step)
        self.fitted_ = False

    def find_node(self, name:str) -> Union[List[TransformerStep], TransformerStep, None]:
        visited = set([])
        matches = []
        Q = copy(self.root_nodes)
        while len(Q) > 0:
            node = Q.pop()
            print(node)
            if node.name == name:
                matches.append(node)

            if node in visited:
                continue

            visited.add(node)
            if node.next is None:
                continue            
            Q.append(node.next)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            return matches
        else:
            return None
                

    def fit(self, dataset: Union[AbstractTimeSeriesDataset, pd.DataFrame]):
        if isinstance(dataset, pd.DataFrame):
            dataset = [dataset]
        dataset_size = len(dataset)
        visited = set()

        transformed_cache = {}
        for r in self.root_nodes:
            # Cache for each node their inmediate previous state.
            # The lenght of the array must be 1 for the regular nodes
            # And a number greater than 1 for the concatenate node
            transformed_cache[r] = [[life for life in dataset]]

        Q = copy(self.root_nodes)

        while len(Q) > 0:
            node = Q.pop(0)
            if node in visited:
                continue

            if isinstance(node, TransformerStep):
                assert(len(transformed_cache[node])==1)
                previous_path_state = transformed_cache[node][0]
                for life in previous_path_state:
                    node.partial_fit(life)
                
                
                
                for i, life in enumerate(previous_path_state):
                    previous_path_state[i] = node.transform(previous_path_state[i])
                if node.next not in transformed_cache:
                     transformed_cache[node.next] = []
                transformed_cache[node.next].append(
                    transformed_cache.pop(node)[0]
                )

            elif isinstance(node, Concatenate):               
                previous_paths_states = transformed_cache[node]
                transformed_cache[node.next] = [[None] * dataset_size]
                for i in range(dataset_size):
                    life_in_previous_paths = [previous_paths_states[j][i] for j in range(len(previous_paths_states))]
                    transformed_cache[node.next][0][i] = node.transform(life_in_previous_paths)
                transformed_cache.pop(node)

            visited.add(node)
            if node.next is not None:
                Q.append(node.next)
        self.fitted_ = True
        self.column_names = transformed_cache[None][0][0].columns
        return self

    def transform(self, df: pd.DataFrame):


        visited = set()
        transformed_cache = {}
        for r in self.root_nodes:
            # Cache for each node their inmediate previous state.
            # The lenght of the array must be 1 for the regular nodes
            # And a number greater than 1 for the concatenate node
            transformed_cache[r] = [df]
        Q = copy(self.root_nodes)

        while len(Q) > 0:
            node = Q.pop(0)
            if node in visited:
                continue

            if isinstance(node, TransformerStep):
                assert(len(transformed_cache[node])==1)
                previous_path_state = transformed_cache[node]
                
                
                previous_path_state[0] = node.transform(previous_path_state[0])
                if node.next not in transformed_cache:
                     transformed_cache[node.next] = []
                transformed_cache[node.next].append(
                    transformed_cache.pop(node)[0]
                )


            elif isinstance(node, Concatenate):                
                previous_paths_states = transformed_cache[node]
                transformed_cache[node.next] =[None]
                
                life_in_previous_paths = [previous_paths_states[j] for j in range(len(previous_paths_states))]
                transformed_cache[node.next][0] = node.transform(life_in_previous_paths)
                transformed_cache.pop(node)



            visited.add(node)
            if node.next is not None:
                Q.append(node.next)
        self.fitted_ = True
        
        return transformed_cache[None][0]