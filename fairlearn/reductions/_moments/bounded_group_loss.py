from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fairlearn.utils._input_validation import _validate_and_reformat_input

from .moment import _ALL, _GROUP_ID, _LABEL, _LOSS, _PREDICTION, LossMoment


class ConditionalLossMoment(LossMoment):

    def __init__(self, loss, *, upper_bound: float | None = None, no_groups: bool = False):
        super().__init__(loss)
        self.upper_bound = upper_bound
        self.no_groups = no_groups

    @property
    def index(self) -> pd.Index:
        """Return the index listing the constraints."""
        return self._index

    def default_objective(self) -> MeanLoss:
        pass

    def load_data(self, X, y, *, sensitive_features) -> None:
        pass

    def gamma(self, predictor: Callable) -> pd.Series:
        pass

    def bound(self) -> pd.Series:
        pass

    def project_lambda(self, lambda_vec: pd.Series) -> pd.Series:
        pass

    def signed_weights(self, lambda_vec: pd.Series | None = None) -> pd.Series:
        pass


ConditionalLossMoment.__module__ = "fairlearn.reductions"


class MeanLoss(ConditionalLossMoment):

    def __init__(self, loss):
        super().__init__(loss, upper_bound=None, no_groups=True)


MeanLoss.__module__ = "fairlearn.reductions"


class BoundedGroupLoss(ConditionalLossMoment):

    def __init__(self, loss, *, upper_bound=None):
        super().__init__(loss, upper_bound=upper_bound, no_groups=False)


class SquareLoss:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = (max_val - min_val) ** 2

    def eval(self, y_true, y_pred):
        pass


class AbsoluteLoss:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = np.abs(max_val - min_val)

    def eval(self, y_true, y_pred):
        pass


AbsoluteLoss.__module__ = "fairlearn.reductions"


class ZeroOneLoss(AbsoluteLoss):

    def __init__(self):
        super().__init__(0, 1)
