

import sys as _sys

from ._base_metrics import count  # noqa: F401
from ._base_metrics import false_negative_rate  # noqa: F401
from ._base_metrics import false_positive_rate  # noqa: F401
from ._base_metrics import mean_prediction  # noqa: F401
from ._base_metrics import selection_rate  # noqa: F401
from ._base_metrics import true_negative_rate  # noqa: F401
from ._base_metrics import true_positive_rate  # noqa: F401
from ._fairness_metrics import demographic_parity_difference  # noqa: F401
from ._fairness_metrics import demographic_parity_ratio  # noqa: F401
from ._fairness_metrics import equal_opportunity_difference  # noqa: F401
from ._fairness_metrics import equal_opportunity_ratio  # noqa: F401
from ._fairness_metrics import equalized_odds_difference  # noqa: F401
from ._fairness_metrics import equalized_odds_ratio  # noqa: F401
from ._generated_metrics import _generated_metric_dict
from ._make_derived_metric import make_derived_metric  # noqa: F401
from ._metric_frame import MetricFrame  # noqa: F401
from ._plot_model_comparison import plot_model_comparison  # noqa: F401
from ._roc_auc import plot_roc_curve_by_group  # noqa: F401

_module_obj = _sys.modules[__name__]
for _name, _func in _generated_metric_dict.items():
    setattr(_module_obj, _name, _func)


_core = ["MetricFrame", "make_derived_metric", "plot_model_comparison", "plot_roc_curve_by_group"]

_fairness = [
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "equalized_odds_ratio",
    "equal_opportunity_difference",
    "equal_opportunity_ratio",
]

_base_metrics = [
    "true_positive_rate",
    "true_negative_rate",
    "false_positive_rate",
    "false_negative_rate",
    "mean_prediction",
    "selection_rate",
    "count",
]

__all__ = _core + _fairness + _base_metrics + list(sorted(_generated_metric_dict.keys()))
