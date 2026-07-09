
import logging
import warnings
from math import ceil
from time import time

from numpy import arange, argmax, unique, zeros
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    is_classifier,
)
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.utils import check_scalar
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    check_consistent_length,
    check_is_fitted,
    check_random_state,
    validate_data,
)

from ._backend_engine import BackendEngine
from ._constants import (
    _CALLBACK_RETURNS_ERROR,
    _IMPORT_ERROR_MESSAGE,
    _KWARG_ERROR_MESSAGE,
    _PREDICTION_FUNCTION_AMBIGUOUS,
    _PROGRESS_UPDATE,
)
from ._preprocessor import FloatTransformer
from ._pytorch_engine import PytorchEngine
from ._tensorflow_engine import TensorflowEngine

logger = logging.getLogger(__name__)


class _AdversarialFairness(BaseEstimator):

    def __init__(
        self,
        *,
        backend="auto",
        predictor_model=None,
        adversary_model=None,
        predictor_loss="auto",
        adversary_loss="auto",
        predictor_function="auto",
        threshold_value=0.5,
        predictor_optimizer="Adam",
        adversary_optimizer="Adam",
        constraints="demographic_parity",
        y_transform="auto",
        sf_transform="auto",
        learning_rate=0.001,
        alpha=1.0,
        epochs=1,
        batch_size=32,
        max_iter=-1,
        shuffle=False,
        progress_updates=None,
        skip_validation=False,
        callbacks=None,
        cuda=None,
        warm_start=False,
        random_state=None,
    ):
        """Initialize class by only storing (kw)args, as per sklearn API."""
        self.backend = backend
        self.predictor_model = predictor_model
        self.adversary_model = adversary_model
        self.predictor_loss = predictor_loss
        self.adversary_loss = adversary_loss
        self.predictor_function = predictor_function
        self.threshold_value = threshold_value
        self.predictor_optimizer = predictor_optimizer
        self.adversary_optimizer = adversary_optimizer
        self.constraints = constraints
        self.y_transform = y_transform
        self.sf_transform = sf_transform
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.progress_updates = progress_updates
        self.skip_validation = skip_validation
        self.callbacks = callbacks
        self.cuda = cuda
        self.warm_start = warm_start
        self.random_state = random_state

    def __setup(self, X, y, A):
        pass

    def fit(self, X, y, *, sensitive_features=None):
        pass

    def partial_fit(self, X, y, *, classes=None, sensitive_features=None):
        pass

    def _raw_predict(self, X):
        pass

    def predict(self, X):
        pass

    def _validate_input(self, X, y, A, reinitialize=False):
        pass

    def _validate_backend(self):
        pass


    def _set_predictor_function(self):
        pass

    def __sklearn_is_fitted__(self):
        """Speed up check_is_fitted."""
        return hasattr(self, "_is_setup")


class AdversarialFairnessClassifier(ClassifierMixin, _AdversarialFairness):

    def __init__(
        self,
        *,
        backend="auto",
        predictor_model=None,
        adversary_model=None,
        predictor_optimizer="Adam",
        adversary_optimizer="Adam",
        constraints="demographic_parity",
        learning_rate=0.001,
        alpha=1.0,
        epochs=1,
        batch_size=32,
        shuffle=False,
        progress_updates=None,
        skip_validation=False,
        callbacks=None,
        cuda=None,
        warm_start=False,
        random_state=None,
    ):
        """Initialize model by setting the predictor loss and function."""
        super(AdversarialFairnessClassifier, self).__init__(
            backend=backend,
            predictor_model=predictor_model,
            adversary_model=adversary_model,
            predictor_loss="classification",
            adversary_loss="auto",
            predictor_function="classification",
            threshold_value=0.5,
            predictor_optimizer=predictor_optimizer,
            adversary_optimizer=adversary_optimizer,
            constraints=constraints,
            learning_rate=learning_rate,
            alpha=alpha,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            progress_updates=progress_updates,
            skip_validation=skip_validation,
            callbacks=callbacks,
            cuda=cuda,
            warm_start=warm_start,
            random_state=random_state,
        )


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if tags.classifier_tags:
            tags.classifier_tags.poor_score = True
        return tags


class AdversarialFairnessRegressor(RegressorMixin, _AdversarialFairness):

    def __init__(
        self,
        *,
        backend="auto",
        predictor_model=None,
        adversary_model=None,
        predictor_optimizer="Adam",
        adversary_optimizer="Adam",
        constraints="demographic_parity",
        learning_rate=0.001,
        alpha=1.0,
        epochs=1,
        batch_size=32,
        shuffle=False,
        progress_updates=None,
        skip_validation=False,
        callbacks=None,
        cuda=None,
        warm_start=False,
        random_state=None,
    ):
        """Initialize model by setting the predictor loss and function."""
        super(AdversarialFairnessRegressor, self).__init__(
            backend=backend,
            predictor_model=predictor_model,
            adversary_model=adversary_model,
            predictor_loss="continuous",
            adversary_loss="auto",
            predictor_function=None,
            predictor_optimizer=predictor_optimizer,
            adversary_optimizer=adversary_optimizer,
            y_transform=None,
            constraints=constraints,
            learning_rate=learning_rate,
            alpha=alpha,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            progress_updates=progress_updates,
            skip_validation=skip_validation,
            callbacks=callbacks,
            cuda=cuda,
            warm_start=warm_start,
            random_state=random_state,
        )


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if tags.regressor_tags:
            tags.regressor_tags.poor_score = True
        return tags
