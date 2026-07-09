
from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from time import time

import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn import clone
from sklearn.dummy import DummyClassifier

from fairlearn.reductions._moments import ClassificationMoment
from fairlearn.reductions._moments.moment import Moment
from fairlearn.utils._common import _filter_kwargs

from ._constants import _INDENTATION, _LINE, _PRECISION

logger = logging.getLogger(__name__)


_MESSAGE_BAD_OBJECTIVE = (
    "Objective needs to be of the same type as constraints. Objective is {}, constraints are {}."
)


class _PredictorAsCallable:
    def __init__(self, classifier):
        self._classifier = classifier

    def __call__(self, X):
        return self._classifier.predict(X)


class _Lagrangian:

    def __init__(
        self,
        *,
        X,
        y,
        estimator,
        constraints: Moment,
        B: float,
        objective: Moment | None = None,
        opt_lambda: bool = True,
        sample_weight_name: str = "sample_weight",
        **kwargs,
    ):
        self.constraints = copy.deepcopy(constraints)
        self.constraints.load_data(X, y, **kwargs)
        if objective is None:
            self.obj = self.constraints.default_objective()
        elif objective._moment_type() == constraints._moment_type():
            self.obj = objective
        else:
            raise ValueError(
                _MESSAGE_BAD_OBJECTIVE.format(objective._moment_type(), constraints._moment_type())
            )
        filtered_kwargs = _filter_kwargs(func=self.obj.load_data, kwargs=kwargs)
        self.obj.load_data(X, y, **filtered_kwargs)
        self.estimator = estimator
        self.B = B
        self.opt_lambda = opt_lambda

        self.hs = pd.Series(dtype="object")

        self.predictors = pd.Series(dtype="object")
        self.errors = pd.Series(dtype="float64")
        self.gammas = pd.DataFrame()
        self.lambdas = pd.DataFrame()
        self.n_oracle_calls = 0
        self.oracle_execution_times = []
        self.n_oracle_calls_dummy_returned = 0
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None
        self.sample_weight_name = sample_weight_name

    def _eval(
        self, Q: pd.Series | Callable, lambda_vec: pd.Series
    ) -> tuple[float, float, pd.Series, float]:
        pass

    def eval_gap(self, Q: pd.Series, lambda_hat: pd.Series, nu: float) -> _GapResult:
        pass



    def best_h(self, lambda_vec: pd.Series) -> tuple[_PredictorAsCallable, int]:
        pass


class _GapResult:

    def __init__(self, L: float, L_low: float, L_high: float, gamma: pd.Series, error: float):
        self.L = L
        self.L_low = L_low
        self.L_high = L_high
        self.gamma = gamma
        self.error = error

