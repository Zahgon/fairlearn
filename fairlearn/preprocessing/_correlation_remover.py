
from collections.abc import Iterable

import narwhals.stable.v1 as nw
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class CorrelationRemover(TransformerMixin, BaseEstimator):

    def __init__(self, *, sensitive_feature_ids: Iterable = (), alpha: float = 1):
        self.sensitive_feature_ids = sensitive_feature_ids
        self.alpha = alpha

    def _split_X(self, X):
        """Split up X into a sensitive and non-sensitive group."""
        sensitive = [self.lookup_[i] for i in self.sensitive_feature_ids]
        non_sensitive = [i for i in range(X.shape[1]) if i not in sensitive]
        return X[:, non_sensitive], X[:, sensitive]

    def _create_lookup(self, X):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        """Transform X by applying the correlation remover."""
        check_is_fitted(self, ["beta_", "_n_features_in_", "lookup_", "sensitive_mean_"])

        X = nw.from_native(X, pass_through=True, eager_only=True)
        if isinstance(X, nw.DataFrame):
            X = X.to_numpy()

        X = validate_data(self, X)
        if self._n_features_in_ != X.shape[1]:
            raise ValueError(
                "X has %d features, but %s is expecting %d features as input"
                % (X.shape[1], self.__class__.__name__, self._n_features_in_)
            )

        X_use, X_sensitive = self._split_X(X)
        X_s_center = X_sensitive - self.sensitive_mean_
        X_filtered = X_use - X_s_center.dot(self.beta_)
        X_use = np.atleast_2d(X_use)
        X_filtered = np.atleast_2d(X_filtered)
        return self.alpha * X_filtered + (1 - self.alpha) * X_use

    def _check_sensitive_features_in_X(self, X) -> None:
        pass
