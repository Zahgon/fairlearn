from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fairlearn.utils._input_validation import (
    _MESSAGE_RATIO_NOT_IN_RANGE,
    _validate_and_reformat_input,
)

from .error_rate import ErrorRate
from .moment import _ALL, _EVENT, _GROUP_ID, _LABEL, _SIGN, ClassificationMoment

_UPPER_BOUND_DIFF = "upper_bound_diff"
_LOWER_BOUND_DIFF = "lower_bound_diff"
_MESSAGE_INVALID_BOUNDS = "Only one of difference_bound and ratio_bound can be used."
_DEFAULT_DIFFERENCE_BOUND = 0.01

_CTRL_EVENT_FORMAT = "control={0},{1}"






class UtilityParity(ClassificationMoment):

    def __init__(
        self,
        *,
        difference_bound: float | None = None,
        ratio_bound: float | None = None,
        ratio_bound_slack: float = 0.0,
    ):
        """Initialize with the ratio value."""
        super(UtilityParity, self).__init__()
        if (difference_bound is None) and (ratio_bound is None):
            self.eps = _DEFAULT_DIFFERENCE_BOUND
            self.ratio = 1.0
        elif (difference_bound is not None) and (ratio_bound is None):
            self.eps = difference_bound
            self.ratio = 1.0
        elif (difference_bound is None) and (ratio_bound is not None):
            self.eps = ratio_bound_slack
            if not (0 < ratio_bound <= 1):
                raise ValueError(_MESSAGE_RATIO_NOT_IN_RANGE)
            self.ratio = ratio_bound
        else:
            raise ValueError(_MESSAGE_INVALID_BOUNDS)

    @property
    def index(self) -> pd.MultiIndex:
        """Return the multi-index listing the constraints."""
        return self._index

    def default_objective(self) -> ErrorRate:
        pass

    def load_data(
        self,
        X,
        y: pd.Series,
        *,
        sensitive_features: pd.Series,
        event: pd.Series | None = None,
        utilities=None,
    ) -> None:
        pass

    def gamma(self, predictor: Callable) -> pd.Series:
        pass

    def bound(self) -> pd.Series:
        pass

    def project_lambda(self, lambda_vec: pd.Series) -> pd.Series:
        pass

    def signed_weights(self, lambda_vec: pd.Series) -> pd.Series:
        pass


UtilityParity.__module__ = "fairlearn.reductions"


class DemographicParity(UtilityParity):

    short_name = "DemographicParity"

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        pass


class TruePositiveRateParity(UtilityParity):

    short_name = "TruePositiveRateParity"

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        pass


class FalsePositiveRateParity(UtilityParity):

    short_name = "FalsePositiveRateParity"

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        pass


class EqualizedOdds(UtilityParity):

    short_name = "EqualizedOdds"

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        pass


class ErrorRateParity(UtilityParity):

    short_name = "ErrorRateParity"

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        pass
