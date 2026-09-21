"""Dixon-Coles tau correction for low-scoring match outcomes.

The tau factor adjusts the independent Poisson probabilities for the four
low-score cells (0-0, 0-1, 1-0, 1-1), capturing the empirical dependence
between home and away goals that a bivariate Poisson model misses.

Reference: Dixon & Coles (1997), Table 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def tau(
    home_goals: int | NDArray[np.int_],
    away_goals: int | NDArray[np.int_],
    lambda_home: float | NDArray[np.float64],
    lambda_away: float | NDArray[np.float64],
    rho: float,
) -> NDArray[np.float64]:
    """Compute the Dixon-Coles tau correction factor.

    Cases (exact from the 1997 paper):
        (0, 0): 1 - lambda_h * lambda_a * rho
        (0, 1): 1 + lambda_h * rho
        (1, 0): 1 + lambda_a * rho
        (1, 1): 1 - rho
        else:   1.0
    """
    home_goals = np.atleast_1d(np.asarray(home_goals, dtype=np.int_))
    away_goals = np.atleast_1d(np.asarray(away_goals, dtype=np.int_))
    lambda_home = np.atleast_1d(np.asarray(lambda_home, dtype=np.float64))
    lambda_away = np.atleast_1d(np.asarray(lambda_away, dtype=np.float64))

    result = np.ones_like(lambda_home, dtype=np.float64)

    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)

    result[m00] = 1.0 - lambda_home[m00] * lambda_away[m00] * rho
    result[m01] = 1.0 + lambda_home[m01] * rho
    result[m10] = 1.0 + lambda_away[m10] * rho
    result[m11] = 1.0 - rho

    return result
