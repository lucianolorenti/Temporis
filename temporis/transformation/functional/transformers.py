"""

The transformer is a high-level class that hold at least two transformation pipelines
* One related to the transformation of the input of the model
* The other related to the target of the model.

Allows accessing the information of the transformed data and is the object that uses the 
dataset iterators to transform the data before feeding it to the model.
"""
import copy
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.algorithms import isin
from sklearn.utils.validation import check_is_fitted
from temporis.transformation.functional.concatenate import Concatenate
from temporis.transformation.functional.pipeline import TemporisPipeline
from temporis.transformation.functional.transformerstep import TransformerStep


logger = logging.getLogger(__name__)

RESAMPLER_STEP_NAME = "resampler"


def transformer_info(transformer):
    """Obtains the transformer information in a serializable format

    Parameters
    ----------
    transformer : Union[LivesPipeline, PandasFeatureUnion, TransformerMixin]
        The transformer step, or pipeline to obtain their underlying information

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        If the transformer passed as an argument doesn't have
        the get_params method.
    """
    if isinstance(transformer, TemporisPipeline):
        return [(name, transformer_info(step)) for name, step in transformer.steps]
    elif isinstance(transformer, PandasFeatureUnion):
        return [
            ("name", "FeatureUnion"),
            (
                "steps",
                [
                    (name, transformer_info(step))
                    for name, step in transformer.transformer_list
                ],
            ),
            ("transformer_weights", transformer.transformer_weights),
        ]

    elif hasattr(transformer, "get_params"):
        d = transformer.get_params()
        d.update({"name": type(transformer).__name__})
        return [(k, d[k]) for k in sorted(d.keys())]
    elif isinstance(transformer, str) and transformer == "passthrough":
        return transformer
    else:
        logger.error(type(transformer))

        raise ValueError("Pipeline elements must have the get_params method")


class Transformer:
    """Transform each life

    The transformer class is the highest level class of the transformer API.
    It contains Transformation Pipelines for the input data and the target,
    and provides mechanism to inspect the structure of the transformed data.

    Parameters
    ----------
    transformerX : LivesPipeline
        Transformer that will be applied to the life data
    transformerY : LivesPipeline
        Transformer that will be applied to the target.
    transformerMetadata : Optional[LivesPipeline], optional
        Transformer that will be used to extract additional
        data from the lives information, by default None
    """

    def __init__(
        self,
        transformerX: Union[TemporisPipeline, TransformerStep],
        transformerY: Union[TemporisPipeline, TransformerStep],
        transformerMetadata: Optional[Union[TemporisPipeline, TransformerStep]] = None,
    ):
        def ensure_pipeline(x):
            if isinstance(x, TemporisPipeline):
                return x
            return TemporisPipeline(x)

        self.transformerX = ensure_pipeline(transformerX)
        self.transformerY = ensure_pipeline(transformerY)
        self.transformerMetadata = (
            ensure_pipeline(transformerMetadata)
            if transformerMetadata is not None
            else None
        )
        self.features = None
        self.fitted_ = False

    def _process_selected_features(self):
        if self.transformerX["selector"] is not None:
            selected_columns = self.transformerX["selector"].get_support(indices=True)
            self.features = [self.features[i] for i in selected_columns]

    def clone(self):
        return copy.deepcopy(self)

    def fit(self, dataset, show_progress: bool = False):
        """Fit the transformer with a given dataset.

        The transformer will fit the X transformer,
        the Y transformer and the metadata transformer

        Parameters
        ----------
        dataset : AbstractLivesDataset
            Dataset

        Returns
        -------
        self
        """
        logger.debug("Fitting Transformer")

        self.transformerX.fit(dataset, show_progress=show_progress)
        self.transformerY.fit(dataset, show_progress=show_progress)
        if self.transformerMetadata is not None:
            self.transformerMetadata.fit(dataset)

        self.minimal_df = dataset[0].head(n=20)
        X = self.transformerX.transform(self.minimal_df)
        self.number_of_features_ = X.shape[1]
        self.fitted_ = True
        self.column_names = self._compute_column_names()
        return self

    def transform(self, life: pd.DataFrame):
        """Transform a life and obtain the input data, the target and the metadata

        Parameters
        ----------
        life : pd.DataFrame
            A life in a form of a DataFrame

        Returns
        -------
        Tuple[np.array, np.array, np.array]
            * The first element consists of the input transformed
            * The second element consits of the target transformed
            * The third element consists of the metadata
        """
        check_is_fitted(self, "fitted_")
        return (
            self.transformX(life),
            self.transformY(life),
            self.transformMetadata(life),
        )

    def transformMetadata(self, df: pd.DataFrame) -> Optional[any]:
        if self.transformerMetadata is not None:
            return self.transformerMetadata.transform(df)
        else:
            return None

    def transformY(self, life: pd.DataFrame) -> np.array:
        """Get the transformed target from a life

        Parameters
        ----------
        life : pd.DataFrame
            A life in a form of a DataFrame

        Returns
        -------
        np.array
            Target obtained from the life
        """
        return self.transformerY.transform(life)

    def transformX(self, life: pd.DataFrame) -> np.array:
        """Get the transformer input data

        Parameters
        ----------
        life : pd.DataFrame
            A life i an form of a DataFrame

        Returns
        -------
        np.array
            Input data transformed
        """
        return self.transformerX.transform(life)

    def columns(self) -> List[str]:
        """Columns names after transformation

        Returns
        -------
        List[str]
        """
        return self.column_names

    @property
    def n_features(self) -> int:
        """Number of features after transformation

        Returns
        -------
        int
        """
        return self.number_of_features_

    def _compute_column_names(self):
        return self.transformerX.column_names

    def description(self):
        return {
            "features": self.features,
            "transformerX": transformer_info(self.transformerX),
            "transformerY": transformer_info(self.transformerY),
        }

    def __str__(self):
        return str(self.description())


def TransformerIdentity(rul_column: str = "RUL") -> Transformer:
    """Return the Transformer

    Parameters
    ----------
    rul_column : str, default, RUL
        Name of the RUL Column

    Returns
    -------
    Transformer
        [description]
    """
    from temporis.transformation.features.selection import ByNameFeatureSelector
    from temporis.transformation.utils import IdentityTransformerStep

    return Transformer(IdentityTransformerStep(), ByNameFeatureSelector([rul_column]))
