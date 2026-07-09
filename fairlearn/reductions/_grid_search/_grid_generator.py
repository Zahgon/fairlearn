
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRID_DIMENSION_WARN_THRESHOLD = 4
GRID_DIMENSION_WARN_TEMPLATE = (
    "The grid has %d dimensions. It is not recommended to use more "
    "than %d, otherwise a prohibitively large grid size is required "
    "to explore the space thoroughly. For such cases consider using "
    "ExponentiatedGradient from the fairlearn.reductions module."
)
GRID_SIZE_WARN_TEMPLATE = (
    "Generating a grid with %d grid points. It is recommended to use at "
    "least %d grid points. Please consider increasing grid_size."
)


class _GridGenerator:

    def __init__(
        self,
        grid_size,
        grid_limit,
        pos_basis,
        neg_basis,
        neg_allowed,
        force_L1_norm,
        grid_offset=None,
    ):
        """Initialize with grid generator utility.

        The `grid_size` is the number of columns to be generated in the grid.
        :type grid_size: int

        The `grid_limit` is the range of the values in the grid generated.
        :type grid_limit: float

        The `neg_allowed` ensures if we want to include negative values in the grid or not.
        If True, the range is doubled.
        :type neg_allowed: boolean

        The `force_L1_norm`, if True, ensures that all points of the grid have the L1 norm equal
        to grid_limit. If False, then grid consists of points whose L1 norm is less than equal to
        grid_limit.
        :type force_L1_norm: boolean

        The `grid_offset` shifts the whole grid by that value.
        :type grid_offset: :class:`pandas:pandas.DataFrame`
        """
        self.dim = len(pos_basis.columns)
        self.neg_allowed = neg_allowed
        self.force_L1_norm = force_L1_norm
        if grid_offset is None:
            self.grid_offset = pd.Series(0, index=pos_basis.index)
        else:
            self.grid_offset = grid_offset

        if self.force_L1_norm:
            true_dim = self.dim - 1
        else:
            true_dim = self.dim

        if true_dim > GRID_DIMENSION_WARN_THRESHOLD:
            logger.warning(GRID_DIMENSION_WARN_TEMPLATE, true_dim, GRID_DIMENSION_WARN_THRESHOLD)

        recommended_min_grid_size = 2**true_dim
        if grid_size < recommended_min_grid_size:
            logger.warning(GRID_SIZE_WARN_TEMPLATE, grid_size, recommended_min_grid_size)

        n_units = (float(grid_size) / (2.0 ** neg_allowed.sum())) ** (1.0 / true_dim) - 1
        n_units = int(np.floor(n_units))
        if n_units < 0:
            n_units = 0

        while True:
            int_grid = self.build_integer_grid(n_units)
            if len(int_grid) >= grid_size:
                pos_coefs = pd.DataFrame(self.accumulator[:grid_size]).T * (
                    float(grid_limit) / n_units
                )
                neg_coefs = -pos_coefs.copy()
                pos_coefs[pos_coefs < 0] = 0.0
                neg_coefs[neg_coefs < 0] = 0.0
                _grid = pos_basis.dot(pos_coefs) + neg_basis.dot(neg_coefs)
                self.grid = _grid.add(self.grid_offset, axis="index")
                break
            n_units = n_units + 1


