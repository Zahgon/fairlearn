from __future__ import annotations

import narwhals.stable.v1 as nw

_SERIES_NAME_NOT_STRING = "Series name must be a string. Value '{0}' was of type {1}"


class GroupFeature:

    def __init__(self, base_name: str, feature_vector, index: int):
        """Help with the metrics."""
        nw_feature_vector = nw.from_native(feature_vector, pass_through=True, allow_series=True)
        is_nw_series = isinstance(nw_feature_vector, nw.Series)

        self.raw_feature_ = (
            list(nw_feature_vector) if not is_nw_series else nw_feature_vector.to_list()
        )

        self.name_ = f"{base_name}{index}"

        if is_nw_series and (name_ := nw_feature_vector.name) is not None:
            if not isinstance(name_, str):
                raise ValueError(_SERIES_NAME_NOT_STRING.format(name_, type(name_)))
            self.name_ = name_
