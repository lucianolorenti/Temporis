import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from temporis.dataset.ts_dataset import AbstractTimeSeriesDataset
import antropy as ant
from itertools import combinations

logger = logging.getLogger(__name__)


def numerical_features(ds: AbstractTimeSeriesDataset) -> pd.DataFrame:
    features = ds.numeric_features()
    data = {c: {"minimum": [], "maximum": [], 'mean':[]} for c in features}
    for life in ds:
        for c in features:
            data[c]['minimum'].append(life[c].min())
            data[c]['maximum'].append(life[c].max())
            data[c]['mean'].append(life[c].mean())
    for k in data.keys():
        data[k]['minimum'] = [np.mean(data[k]['minimum'])]
        data[k]['maximum'] = [np.mean(data[k]['maximum'])]
        data[k]['mean'] = [np.mean(data[k]['mean'])]
    df= pd.DataFrame(data)

    return df.T

