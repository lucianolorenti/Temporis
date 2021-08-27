from typing import List, Optional, Union

from numpy.lib.arraysetops import isin


class TransformerStepMixin:
    def __init__(self, name:Optional[str]=None):
        self.name_ = name
        self.previous = []
        self.next = None

    @property
    def name(self):
        if self.name_ is not None:
            return self.name_
        else:
            return self.__class__.__name__

    def __call__(self, prev: Union[List["TransformerStepMixin"], "TransformerStepMixin"]):
        if not isinstance(prev, list):
            prev = [prev]
        self.previous.extend(prev)
        for p in prev:
            p.next = self
        return self

