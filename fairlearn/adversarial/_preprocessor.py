
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target


class FloatTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, transformer="auto"):
        """Initialize empty transformers with the given distribution assumption.

        Parameters
        ----------
        transformer : str, sklearn.base.TransformerMixin, optional, default = "auto"
            This is a string that indicates the transformer, such as
            :code:`"auto"`, :code:`"one_hot_encoder"`, :code:`"binarizer"`.
            Or, None, for pass-through. Or, a transformer object.
        """
        self.transformer = transformer

    def _check(self, X, dtype=None, init=False):
        """
        Check X and convert to 2d ndarray.

        dtype : numpy.dtype
            None to keep dtypes, float to coerce to numeric.

        init : bool
            Whether this is the first call to _check or not. Useful to store
            the dimensions of X, so we can use this for inverse_transform

        Returns
        -------
        X : numpy.ndarray
            validated input
        """
        X = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=dtype,
            ensure_2d=False,
        )
        if init:
            self.input_dim_ = X.ndim
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        """Transform X using the fitted encoder or passthrough."""
        if isinstance(self.transformer, str) or self.transformer is None:
            return (
                self.transform_.transform(self._check(X)).astype(float)
                if self.inferred_type_ in ["binary", "multiclass"]
                else self._check(X, dtype=float)
            )
        else:
            return self.transform_.transform(X)

    def inverse_transform(self, y):
        pass
