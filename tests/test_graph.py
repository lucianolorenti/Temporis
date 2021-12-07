from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.transformation import Concatenate as TransformationConcatenate
from temporis.transformation import Transformer
from temporis.transformation.features.imputers import PerColumnImputer
from temporis.transformation.features.outliers import IQROutlierRemover
from temporis.transformation.features.scalers import MinMaxScaler
from temporis.transformation.features.selection import ByNameFeatureSelector
from temporis.transformation.features.transformation import MeanCentering
from temporis.transformation.functional.graph_utils import (
    root_nodes,
    topological_sort_iterator,
)
from temporis.transformation.functional.pipeline import TemporisPipeline, make_pipeline
from temporis.transformation.functional.transformerstep import TransformerStep


class VisitableNode(TransformerStep):
    def visit(self):
        pass


class A(VisitableNode):
    pass


class B(VisitableNode):
    class B1(VisitableNode):
        pass

    class B2(VisitableNode):
        pass

    def __init__(self, name):
        super().__init__(name)
        self.b1 = B.B1("B1")
        self.b2 = B.B2("B2")(self.b1)
        from typing import Any, List, Optional, Tuple, Union


from copy import deepcopy

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
from temporis.transformation import Concatenate as TransformationConcatenate
from temporis.transformation import Transformer
from temporis.transformation.features.imputers import PerColumnImputer
from temporis.transformation.features.outliers import IQROutlierRemover
from temporis.transformation.features.scalers import MinMaxScaler
from temporis.transformation.features.selection import ByNameFeatureSelector
from temporis.transformation.features.transformation import MeanCentering
from temporis.transformation.functional.graph_utils import (
    root_nodes,
    topological_sort_iterator,
)
from temporis.transformation.functional.pipeline import TemporisPipeline, make_pipeline
from temporis.transformation.functional.transformerstep import TransformerStep


class VisitableNode(TransformerStep):
    def visit(self):
        pass


class A(VisitableNode):
    pass


class B(VisitableNode):
    class B1(VisitableNode):
        pass

    class B2(VisitableNode):
        pass

    def __init__(self, name):
        super().__init__(name)
        self.b1 = B.B1("B1")
        self.b2 = B.B2("B2")(self.b1)

    def visit(self):
        for n in self.next:
            self.disconnect(n)
            self.b2.add_next(n)
        self.b1(self)


class C(VisitableNode):
    pass


class Node(TransformerStep):
    pass

    def visit(self):
        self.disconnect(self.next)
        self.b2.add_next(self.next)
        self.b1(self)


class C(VisitableNode):
    pass


class Node(TransformerStep):
    pass


class TestGraph:
    def test_simple(self):
        pipe = Node("A")
        pipe = Node("B")(pipe)
        pipe = Node("C")(pipe)

        assert pipe.previous[0].name == "B"
        assert pipe.previous[0].previous[0].name == "A"

        topological_sort_iterator

    def test_graph_updating(self):
        pipe = A("A")
        pipe = B("B")(pipe)
        pipe = C("C")(pipe)

        pipe.previous[0].visit()
        assert pipe.previous[0].name == "B2"
        assert pipe.previous[0].previous[0].name == "B1"
        assert pipe.previous[0].previous[0].previous[0].name == "B"

        pipe = A("A")
        pipe = B("B")(pipe)
        pipe = C("C")(pipe)

        result = []
        for a in topological_sort_iterator(pipe):
            a.visit()
            result.append(a.name)

        assert result == ["A", "B", "B1", "B2", "C"]

    def test_diamond(self):
        pipe = Node("A")
        pipeB = Node("B")(pipe)
        pipeC = Node("C")(pipe)
        pipeD = Node("D")(pipe)
        pipe = Node("E")([pipeB, pipeC, pipeD])
