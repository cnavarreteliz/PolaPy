import numpy as np
import pandas as pd


class RangeException(Exception):
    pass


def esteban_ray(
    data: pd.DataFrame,
    pi: str = "pi",
    y: str = "y",
    alpha: float = 0,
    K: float = None
) -> float:
    """
    Polarization coefficient according to the proposal of Esteban and Ray (1994).

    According to the authors, the polarization coefficient is conceptualized from three features:

    "
    FEATURE 1: There must be a high degree of homogeneity within each group.
    FEATURE 2: There must be a high degree of heterogeneity across groups.
    FEATURE 3: There must be a small number of significantly sized groups. In particular, groups of insignificant size (e.g., isolated individuals) carry little weight. 
    "
    Parameters
    ----------
    data : pd.DataFrame
        Dataset to compute the metric.
    pi, y : names of variables in data, optional
        Inputs for interpreting the DataFrame. If no values are specified, they are interpreted as pi and y respectively.
    alpha : float, optional
        Freedom degree, by default 0. According to Esteban and Ray (1994), alpha must be defined in the range [0, 1.6).
    K : float, optional
        Freedom degree, by default None. If K is not defined, the function will compute alpha based on Esteban and Ray.

    Returns
    -------
    float
        The polarization coefficient.

    Raises
    ------
    RangeException
        The alpha parameter is not bounded in the range proposed by Esteban and Ray (1994). The value must be defined in the range [0, 1.6).

    References
    ----------
    Esteban, J. M., & Ray, D. (1994). On the measurement of polarization. Econometrica: Journal of the Econometric Society, 819-851.

    Duclos, J. Y., Esteban, J., & Ray, D. (2006). Polarization: concepts, measurement, estimation. In The Social Economics of Poverty (pp. 54-102). Routledge.

    See Also
    --------
    divisiveness : Divisiveness.

    Notes
    -----

    The polarization coefficient is calculated as follows:

    $
    K \sum\limits_{i=1}^{N} \sum\limits_{j=1}^{N} \pi_{i}^{1+\alpha}\pi_{j}|y_{i}-y_{j}|$
    $

    where $\pi_{i}$ is the mass and $y_{i}$ a given characteristic of $i$. In the seminal paper, the authors use the log of income.

    """

    if alpha < 0 or alpha >= 1.6:
        raise RangeException(
            "The alpha parameter is not bounded in the range proposed by Esteban and Ray (1994). The value must be defined in the range [0, 1.6)"
        )

    weights = data[pi].values
    rates = data[y].values

    if not K:
        K = 1 / (weights.sum() ** (2 + alpha))

    xx = np.multiply.outer(weights ** (1 + alpha), weights)
    yy = np.absolute(np.subtract.outer(rates, rates))

    return K * np.sum(xx * yy)
