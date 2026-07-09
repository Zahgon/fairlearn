from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
    check_is_fitted,
)
from sklearn.calibration import LabelEncoder
from sklearn.dummy import check_random_state
from sklearn.metrics import log_loss
from sklearn.utils.multiclass import type_of_target

from sklearn.utils.validation import validate_data

from fairlearn.utils._input_validation import _validate_and_reformat_input

LOGGER = logging.getLogger(__name__)


class PrototypeRepresentationLearner(ClassifierMixin, TransformerMixin, BaseEstimator):

    n_prototypes: int
    reconstruct_weight: float
    target_weight: float
    fairness_weight: float
    random_state: int | np.random.RandomState | None
    tol: float
    max_iter: int
    coef_: np.ndarray
    n_iter_: int
    n_features_in_: int
    classes_: np.ndarray | None
    _has_target: bool
    _label_encoder: LabelEncoder | None
    _groups: pd.Series | None
    _prototypes_: np.ndarray
    _alpha_: np.ndarray
    _prototype_dim: int
    _prototype_predictions_size: int
    _prototype_vectors_size: int
    _optimizer_size: int

    def __init__(
        self,
        n_prototypes: int = 2,
        reconstruct_weight: float = 1.0,
        target_weight: float = 1.0,
        fairness_weight: float = 1.0,
        random_state: int | np.random.RandomState | None = None,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        self.n_prototypes = n_prototypes
        self.fairness_weight = fairness_weight
        self.reconstruct_weight = reconstruct_weight
        self.target_weight = target_weight
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y=None, *, sensitive_features=None) -> PrototypeRepresentationLearner:
        pass

    def _optimize(
        self, X, y, sensitive_features: pd.Series | None, random_state: np.random.RandomState
    ) -> PrototypeRepresentationLearner:
        pass

    def _objective(self, x: np.ndarray, X, y, sensitive_features: pd.Series | None) -> float:
        pass

    def transform(self, X) -> np.ndarray:
        r"""Transform the input data X using the learned prototype representation.

        Each sample is transformed to its associated learned latent mapping, i.e. the softmax of
        its negative distance to the prototypes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        np.ndarray
            The transformed data.

        Notes
        -----
        This method checks if the model is fitted, validates the input data,
        and then applies the learned prototype representation.
        """
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)

        M = self._get_latent_mapping(X, self._prototypes_, dimension_weights=self._alpha_)
        return M

    def predict_proba(self, X) -> np.ndarray:
        pass

    def predict(self, X) -> np.ndarray:
        pass



    @staticmethod
    def _get_latent_mapping(
        X, prototypes: np.ndarray, dimension_weights: np.ndarray
    ) -> np.ndarray:
        r"""
        Compute the latent mapping of the input data X to the given prototypes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to be mapped.
        prototypes : np.ndarray of shape (n_prototypes, n_features)
            The prototype vectors to which the input data will be mapped.
        dimension_weights : np.ndarray of shape (n_features,)
            The weights for each dimension used in the distance calculation.

        Returns
        -------
        np.ndarray of shape (n_samples, n_prototypes)
            The latent mapping of the input data to the prototypes, where each
            element represents the softmax-transformed negative distance between
            a sample and a prototype.
        """
        distances = cdist(X, prototypes, metric="euclidean", w=dimension_weights)
        M = softmax(-distances, axis=1)
        return M

    def _validate_X_y(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        pass

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.target_tags.required = False
        return tags

