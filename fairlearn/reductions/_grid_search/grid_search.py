
import copy
import logging
from time import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted

from fairlearn.reductions._moments import ClassificationMoment, Moment

from ._grid_generator import _GridGenerator

logger = logging.getLogger(__name__)

TRADEOFF_OPTIMIZATION = "tradeoff_optimization"


class GridSearch(BaseEstimator, MetaEstimatorMixin):

    def __init__(
        self,
        estimator,
        constraints,
        selection_rule=TRADEOFF_OPTIMIZATION,
        constraint_weight=0.5,
        grid_size=10,
        grid_limit=2.0,
        grid_offset=None,
        grid=None,
        sample_weight_name="sample_weight",
    ):
        """Construct a GridSearch object."""
        self.estimator = estimator
        if not isinstance(constraints, Moment):
            raise RuntimeError("Unsupported disparity metric")
        self.constraints = constraints

        if selection_rule == TRADEOFF_OPTIMIZATION:
            if not (0.0 <= constraint_weight <= 1.0):
                raise RuntimeError("Must specify constraint_weight between 0.0 and 1.0")
        else:
            raise RuntimeError("Unsupported selection rule")
        self.selection_rule = selection_rule
        self.constraint_weight = float(constraint_weight)
        self.objective_weight = 1.0 - constraint_weight

        self.grid_size = grid_size
        self.grid_limit = float(grid_limit)
        self.grid_offset = grid_offset
        self.grid = grid
        self.sample_weight_name = sample_weight_name

    def fit(self, X, y, **kwargs):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
