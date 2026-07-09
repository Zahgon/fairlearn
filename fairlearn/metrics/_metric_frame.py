from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length

from fairlearn.utils._input_manipulations import _convert_to_ndarray_and_squeeze

from ._annotated_metric_function import AnnotatedMetricFunction
from ._bootstrap import calculate_pandas_quantiles, generate_bootstrap_samples
from ._disaggregated_result import (
    _INVALID_ERRORS_VALUE_ERROR_MESSAGE,
    _VALID_ERROR_STRING,
    DisaggregatedResult,
)
from ._group_feature import GroupFeature

logger = logging.getLogger(__name__)

_SF_DICT_CONVERSION_FAILURE = (
    "DataFrame.from_dict() failed on sensitive features. "
    "Please ensure each array is strictly 1-D. "
    "The __cause__ field of this exception may contain further information."
)
_FEATURE_LIST_NONSCALAR = "Feature lists must be of scalar types"
_FEATURE_DF_COLUMN_BAD_NAME = "DataFrame column names must be strings. Name '{0}' is of type {1}"
_DUPLICATE_FEATURE_NAME = "Detected duplicate feature name: '{0}'"
_TOO_MANY_FEATURE_DIMS = "Feature array has too many dimensions"
_SAMPLE_PARAMS_NOT_DICT = "Sample parameters must be a dictionary"
_SAMPLE_PARAM_KEYS_NOT_IN_FUNC_DICT = "Keys in 'sample_params' do not match those in 'metric'"

_COMPARE_METHODS = ["between_groups", "to_overall"]
_INVALID_COMPARE_METHOD = "Unrecognised comparison method: {0}"

_BOOTSTRAP_NEED_N_AND_CI = "Must specify both n_boot and ci_quantiles"
_BOOTSTRAP_N_BOOT_INT_GT_ZERO = "Must have n_boot be a positive integer"
_BOOTSTRAP_CI_INVALID = "Must have all ci_quantiles be floats in (0, 1)"
_BOOTSTRAP_NOT_INITIALIZED = (
    "Could not compute confidence intervals:"
    " Bootstrapping parameters n_boot and ci_quantiles were not specified"
    " in the MetricFrame constructor."
)


class MetricFrame:

    def __init__(
        self,
        *,
        metrics: Callable | dict[str, Callable],
        y_true,
        y_pred,
        sensitive_features,
        control_features=None,
        sample_params: dict[str, Any] | dict[str, dict[str, Any]] | None = None,
        n_boot: int | None = None,
        ci_quantiles: list[float] | None = None,
        random_state: int | np.random.RandomState | None = None,
    ):
        """Read a placeholder comment."""
        check_consistent_length(y_true, y_pred)

        y_t = _convert_to_ndarray_and_squeeze(y_true)
        y_p = _convert_to_ndarray_and_squeeze(y_pred)

        all_data = pd.DataFrame.from_dict({"y_true": list(y_t), "y_pred": list(y_p)})

        annotated_funcs = self._get_annotated_metric_functions(metrics, sample_params, all_data)

        sf_list = self._process_features("sensitive_feature_", sensitive_features, y_t)
        self._sf_names = [x.name_ for x in sf_list]

        cf_list = None
        self._cf_names = None
        if control_features is not None:
            cf_list = self._process_features("control_feature_", control_features, y_t)
            self._cf_names = [x.name_ for x in cf_list]

        for sf in sf_list:
            all_data[sf.name_] = sf.raw_feature_
        if cf_list is not None:
            for cf in cf_list:
                all_data[cf.name_] = cf.raw_feature_

        nameset = set()
        namelist = self._sf_names
        if self._cf_names:
            namelist = namelist + self._cf_names
        for name in namelist:
            if name in nameset:
                raise ValueError(_DUPLICATE_FEATURE_NAME.format(name))
            nameset.add(name)

        self._result_cache = dict()

        result = DisaggregatedResult.create(
            data=all_data,
            annotated_functions=annotated_funcs,
            sensitive_feature_names=self._sf_names,
            control_feature_names=self._cf_names,
        )
        self._populate_results(result)

        self._ci_quantiles = ci_quantiles
        self._n_boot = n_boot

        if n_boot is not None and ci_quantiles is not None and len(ci_quantiles) > 0:
            if not isinstance(n_boot, int) or n_boot < 1:
                raise ValueError(_BOOTSTRAP_N_BOOT_INT_GT_ZERO)
            for _ci in ci_quantiles:
                if not isinstance(_ci, float) or _ci <= 0 or _ci >= 1:
                    raise ValueError(_BOOTSTRAP_CI_INVALID)

            _bootstrap_samples = generate_bootstrap_samples(
                n_samples=n_boot,
                random_state=random_state,
                data=all_data,
                annotated_functions=annotated_funcs,
                sensitive_feature_names=self._sf_names,
                control_feature_names=self._cf_names,
            )

            self._populate_results_ci(_bootstrap_samples, ci_quantiles)
        elif (n_boot is not None) ^ ((ci_quantiles is not None) and (len(ci_quantiles)) > 0):
            raise ValueError(_BOOTSTRAP_NEED_N_AND_CI)

    def _extract_result(self, underlying_result, no_control_levels: bool):
        pass

    def _none_to_nan(self, target: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        pass

    def _populate_results(self, raw_result: DisaggregatedResult):
        pass

    def _populate_results_ci(
        self, bootstrap_samples: list[DisaggregatedResult], ci_quantiles: list[float]
    ):
        pass

    @property
    def overall(self) -> Any | pd.Series | pd.DataFrame:
        pass

    @property
    def overall_ci(self) -> list[Any | pd.Series | pd.DataFrame]:
        pass

    @property
    def by_group(self) -> pd.Series | pd.DataFrame:
        pass

    @property
    def by_group_ci(self) -> list[pd.Series] | list[pd.DataFrame]:
        pass

    @property
    def control_levels(self) -> list[str] | None:
        pass

    @property
    def sensitive_levels(self) -> list[str]:
        pass

    @property
    def ci_quantiles(self) -> list[float] | None:
        pass

    @property
    def n_boot(self) -> int | None:
        pass

    def _group(
        self,
        disagg_result: DisaggregatedResult,
        grouping_function: Literal["min", "max"],
        errors: Literal["raise", "coerce"] = "raise",
    ) -> Any | pd.Series | pd.DataFrame:
        pass


    def group_max(
        self, errors: Literal["raise", "coerce"] = "raise"
    ) -> Any | pd.Series | pd.DataFrame:
        pass

    def group_max_ci(self) -> list[Any] | list[pd.Series] | list[pd.DataFrame]:
        pass

    def group_min(
        self, errors: Literal["raise", "coerce"] = "raise"
    ) -> Any | pd.Series | pd.DataFrame:
        pass

    def group_min_ci(self) -> list[Any] | list[pd.Series] | list[pd.DataFrame]:
        pass

    def difference(
        self,
        method: Literal["between_groups", "to_overall"] = "between_groups",
        errors: Literal["raise", "coerce"] = "coerce",
    ) -> Any | pd.Series | pd.DataFrame:
        """Return the maximum absolute difference between groups for each metric.

        This method calculates a scalar value for each underlying metric by
        finding the maximum absolute difference between the entries in each
        combination of sensitive features in the :attr:`.by_group` property.

        Similar to other methods, the result type varies with the
        specification of the metric functions, and whether control features
        are present or not.

        There are two allowed values for the ``method=`` parameter. The
        value ``between_groups`` computes the maximum difference between
        any two pairs of groups in the :attr:`.by_group` property (i.e.
        ``group_max() - group_min()``). Alternatively, ``to_overall``
        computes the difference between each subgroup and the
        corresponding value from :attr:`.overall` (if there are control
        features, then :attr:`.overall` is multivalued for each metric).
        The result is the absolute maximum of these values.

        Read more in the :ref:`User Guide <assessment_compare_harms>`.

        Parameters
        ----------
        method : string {'between_groups', 'to_overall'}, default :code:`between_groups`
            How to compute the aggregate.
        errors : {'raise', 'coerce'}, default :code:`coerce`
            if 'raise', then invalid parsing will raise an exception
            if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
            The exact type follows the table in :attr:`.MetricFrame.overall`.
        """
        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        if method not in _COMPARE_METHODS:
            raise ValueError(_INVALID_COMPARE_METHOD.format(method))

        value = self._result_cache["difference"][method][errors]
        if isinstance(value, Exception):
            raise value
        else:
            return value

    def difference_ci(
        self, method: Literal["between_groups", "to_overall"] = "between_groups"
    ) -> list[Any] | list[pd.Series] | list[pd.DataFrame]:
        pass

    def ratio(
        self,
        method: Literal["between_groups", "to_overall"] = "between_groups",
        errors: Literal["raise", "coerce"] = "coerce",
    ) -> Any | pd.Series | pd.DataFrame:
        """Return the minimum ratio between groups for each metric.

        This method calculates a scalar value for each underlying metric by
        finding the minimum ratio (that is, the ratio is forced to be
        less than unity) between the entries in each
        column of the :attr:`.by_group` property.

        Similar to other methods, the result type varies with the
        specification of the metric functions, and whether control features
        are present or not.

        There are two allowed values for the ``method=`` parameter. The
        value ``between_groups`` computes the minimum ratio between
        any two pairs of groups in the :attr:`.by_group` property (i.e.
        ``group_min() / group_max()``). Alternatively, ``to_overall``
        computes the ratio between each subgroup and the
        corresponding value from :attr:`.overall` (if there are control
        features, then :attr:`.overall` is multivalued for each metric),
        expressing the ratio as a number less than 1.
        The result is the minimum of these values.

        Read more in the :ref:`User Guide <assessment_compare_harms>`.

        Parameters
        ----------
        method : string {'between_groups', 'to_overall'}, default :code:`between_groups`
            How to compute the aggregate.
        errors : {'raise', 'coerce'}, default :code:`coerce`
            if 'raise', then invalid parsing will raise an exception
            if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
            The exact type follows the table in :attr:`.MetricFrame.overall`.
        """
        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        if method not in _COMPARE_METHODS:
            raise ValueError(_INVALID_COMPARE_METHOD.format(method))

        value = self._result_cache["ratio"][method][errors]
        if isinstance(value, Exception):
            raise value
        else:
            return value

    def ratio_ci(
        self, method: Literal["between_groups", "to_overall"] = "between_groups"
    ) -> list[Any] | list[pd.Series] | list[pd.DataFrame]:
        pass

    def _get_annotated_metric_functions(
        self,
        metric: Callable | dict[str, Callable],
        sample_params: dict[str, Any] | dict[str, dict[str, Any]] | None,
        all_data: pd.DataFrame,
    ) -> dict[str, AnnotatedMetricFunction]:
        pass


    def _process_features(self, base_name, features, sample_array) -> list[GroupFeature]:
        pass

    def _check_bootstrap_initialized(self):
        pass
