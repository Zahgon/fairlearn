from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from fairlearn.reductions._moments import ClassificationMoment
from fairlearn.reductions._moments.moment import Moment

from ._constants import (
    _ACCURACY_MUL,
    _INDENTATION,
    _MIN_ITER,
    _PRECISION,
    _REGRET_CHECK_INCREASE_T,
    _REGRET_CHECK_START_T,
    _SHRINK_ETA,
    _SHRINK_REGRET,
)
from ._lagrangian import _Lagrangian

logger = logging.getLogger(__name__)


class ExponentiatedGradient(BaseEstimator, MetaEstimatorMixin):

    def __init__(
        self,
        estimator,
        constraints: Moment,
        *,
        objective: Moment | None = None,
        eps: float = 0.01,
        max_iter: int = 50,
        nu: float | None = None,
        eta0: float = 2.0,
        run_linprog_step: bool = True,
        sample_weight_name: str = "sample_weight",
    ):
        self.estimator = estimator
        self.constraints = constraints
        self.objective = objective
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        self.run_linprog_step = run_linprog_step
        self.sample_weight_name = sample_weight_name

    def fit(self, X, y, **kwargs):
        pass

    def predict(self, X, random_state=None):
        pass

    def _pmf_predict(self, X):
        pass
