
from warnings import warn

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ..utils._common import _get_soft_predictions
from ..utils._input_validation import _validate_and_reformat_input
from ._constants import (
    BASE_ESTIMATOR_NONE_ERROR_MESSAGE,
    BASE_ESTIMATOR_NOT_FITTED_WARNING,
)


class InterpolatedThresholder(MetaEstimatorMixin, BaseEstimator):

    def __init__(self, estimator, interpolation_dict, prefit=False, predict_method="auto"):
        self.estimator = estimator
        self.interpolation_dict = interpolation_dict
        self.prefit = prefit
        self.predict_method = predict_method

    def fit(self, X, y, **kwargs):
        pass

    def _pmf_predict(self, X, *, sensitive_features):
        pass

    def predict(self, X, *, sensitive_features, random_state=None):
        pass
