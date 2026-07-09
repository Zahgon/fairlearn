from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd

from fairlearn.utils._input_validation import _validate_and_reformat_input

from .moment import _ALL, _LABEL, ClassificationMoment

_MESSAGE_BAD_COSTS = (
    "costs needs to be a dictionary with keys "
    "'fp' and 'fn' containing non-negative values, which are not both zero"
)


class ErrorRate(ClassificationMoment):

    def __init__(self, *, costs: dict[Literal["fp", "fn"], float] | None = None):
        """Initialize the costs."""
        super(ErrorRate, self).__init__()
        if costs is None:
            self.fp_cost = 1.0
            self.fn_cost = 1.0
        elif (
            isinstance(costs, dict)
            and costs.keys() == {"fp", "fn"}
            and costs["fp"] >= 0.0
            and costs["fn"] >= 0.0
            and costs["fp"] + costs["fn"] > 0.0
        ):
            self.fp_cost = costs["fp"]
            self.fn_cost = costs["fn"]
        else:
            raise ValueError(_MESSAGE_BAD_COSTS)

    def load_data(self, X, y, *, sensitive_features) -> None:
        pass

    @property
    def index(self) -> pd.Index:
        """Return the index listing the constraints."""
        return self._index

    def gamma(self, predictor: Callable) -> pd.Series:
        pass

    def project_lambda(self, lambda_vec: pd.Series) -> pd.Series:
        pass

    def signed_weights(self, lambda_vec: pd.Series | None = None) -> pd.Series:
        pass
