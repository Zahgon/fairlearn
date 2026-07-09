

from __future__ import annotations

import logging
from typing import Literal
from warnings import warn

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from ..utils._common import _get_soft_predictions
from ..utils._input_validation import _KW_CONTROL_FEATURES, _validate_and_reformat_input
from ._constants import (
    BASE_ESTIMATOR_NONE_ERROR_MESSAGE,
    BASE_ESTIMATOR_NOT_FITTED_WARNING,
    LABEL_KEY,
    OUTPUT_SEPARATOR,
    SCORE_KEY,
    SENSITIVE_FEATURE_KEY,
)
from ._interpolated_thresholder import InterpolatedThresholder
from ._relaxed_constraints import maximize_objective_with_tolerance
from ._tradeoff_curve_utilities import (
    METRIC_DICT,
    _extend_confusion_matrix,
    _interpolate_curve,
    _tradeoff_curve,
)

DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE = "{} need to be of equal length."
NON_BINARY_LABELS_ERROR_MESSAGE = "Labels other than 0/1 were provided."
MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE = (
    "Post processing currently only supports a single column in {}."
)
SENSITIVE_FEATURE_NAME_CONFLICT_DETECTED_ERROR_MESSAGE = (
    "A sensitive feature named {} or {} "
    "was detected. Please rename your column and try again.".format(SCORE_KEY, LABEL_KEY)
)
SCORES_DATA_TOO_MANY_COLUMNS_ERROR_MESSAGE = "The provided scores data contains multiple columns."
UNEXPECTED_DATA_TYPE_ERROR_MESSAGE = "Unexpected data type {} encountered."

logger = logging.getLogger(__name__)


SIMPLE_CONSTRAINTS = {
    "selection_rate_parity": "selection_rate",
    "demographic_parity": "selection_rate",
    "false_positive_rate_parity": "false_positive_rate",
    "false_negative_rate_parity": "false_negative_rate",
    "true_positive_rate_parity": "true_positive_rate",
    "true_negative_rate_parity": "true_negative_rate",
}

ALL_CONSTRAINTS = list(SIMPLE_CONSTRAINTS.keys()) + ["equalized_odds"]

OBJECTIVES_FOR_SIMPLE_CONSTRAINTS = {
    "selection_rate",
    "true_positive_rate",
    "true_negative_rate",
    "accuracy_score",
    "balanced_accuracy_score",
}


OBJECTIVES_FOR_EQUALIZED_ODDS = {
    "accuracy_score",
    "balanced_accuracy_score",
}

NO_CONTROL_FEATURES = "Control features are not supported by ThresholdOptimizer"
NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE = (
    "Currently only the following constraints are supported: {}.".format(
        ", ".join(sorted(ALL_CONSTRAINTS))
    )
)
NOT_SUPPORTED_OBJECTIVES_FOR_SIMPLE_CONSTRAINTS_ERROR_MESSAGE = (
    "For {{}} only the following objectives are supported: {}.".format(
        ", ".join(sorted(OBJECTIVES_FOR_SIMPLE_CONSTRAINTS))
    )
)
NOT_SUPPORTED_OBJECTIVES_FOR_EQUALIZED_ODDS_ERROR_MESSAGE = (
    "For equalized_odds only the following objectives are supported: {}.".format(
        ", ".join(sorted(OBJECTIVES_FOR_EQUALIZED_ODDS))
    )
)


class ThresholdOptimizer(MetaEstimatorMixin, BaseEstimator):

    def __init__(
        self,
        *,
        estimator=None,
        constraints: Literal[
            "demographic_parity",
            "equalized_odds",
            "false_negative_rate_parity",
            "false_positive_rate_parity",
            "selection_rate_parity",
            "true_negative_rate_parity",
            "true_positive_rate_parity",
        ] = "demographic_parity",
        objective: Literal[
            "accuracy_score",
            "balanced_accuracy_score",
            "selection_rate",
            "true_positive_rate",
            "true_negative_rate",
        ] = "accuracy_score",
        grid_size: int = 1000,
        flip: bool = False,
        prefit: bool = False,
        predict_method: Literal["auto", "predict_proba", "decision_function", "predict"] = "auto",
        tol: float | None = None,
    ):
        self.estimator = estimator
        self.constraints = constraints
        self.objective = objective
        self.grid_size = grid_size
        self.flip = flip
        self.prefit = prefit
        self.predict_method = predict_method
        self.tol = tol

    def fit(self, X, y, *, sensitive_features, **kwargs):
        pass

    def predict(self, X, *, sensitive_features, random_state=None):
        pass

    def _pmf_predict(self, X, *, sensitive_features):
        pass

    def _threshold_optimization_for_simple_constraints(
        self, sensitive_features, labels, scores
    ) -> InterpolatedThresholder:
        pass

    def _threshold_optimization_for_equalized_odds(self, sensitive_features, labels, scores):
        pass


def _reformat_and_group_data(sensitive_features, labels, scores, sensitive_feature_names=None):
    pass


def _reformat_data_into_dict(key, data_dict, additional_data):
    pass
