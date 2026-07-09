from __future__ import annotations

from collections.abc import Callable

import pandas as pd

_GROUP_ID = "group_id"
_EVENT = "event"
_LABEL = "label"
_LOSS = "loss"
_PREDICTION = "pred"
_ALL = "all"
_SIGN = "sign"


class Moment:

    def __init__(self):
        self.data_loaded = False

    def load_data(self, X, y: pd.Series, *, sensitive_features: pd.Series | None = None) -> None:
        pass

    @property
    def total_samples(self) -> int:
        pass

    @property
    def _y_as_series(self) -> pd.Series:
        pass

    @property
    def index(self) -> pd.MultiIndex | pd.Index:
        """Return a pandas (multi-)index listing the constraints."""
        raise NotImplementedError()

    def gamma(self, predictor: Callable) -> pd.Series:
        """Calculate the degree to which constraints are currently violated by the predictor."""
        raise NotImplementedError()

    def bound(self) -> pd.Series:
        """Return vector of fairness bound constraint the length of gamma."""
        raise NotImplementedError()

    def project_lambda(self, lambda_vec: pd.Series) -> pd.Series:
        """Return the projected lambda values."""
        raise NotImplementedError()

    def signed_weights(self, lambda_vec: pd.Series) -> pd.Series:
        """Return the signed weights."""
        raise NotImplementedError()

    def _moment_type(self) -> type[Moment]:
        pass

    def default_objective(self) -> Moment:
        """Return the default objective for the moment."""
        raise NotImplementedError()


Moment.__module__ = "fairlearn.reductions"


class ClassificationMoment(Moment):

    pass


ClassificationMoment.__module__ = "fairlearn.reductions"


class LossMoment(Moment):

    def __init__(self, loss):
        super().__init__()
        self.reduction_loss = loss

    pass


LossMoment.__module__ = "fairlearn.reductions"
