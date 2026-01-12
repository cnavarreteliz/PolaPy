import numpy as np
import pandas as pd


def wang_tsui(
    data: pd.DataFrame,
    K: float = 1.0,
    gamma: float = 0.5,
    value: str = "value",
    rate: str = "rate"
) -> float:
    """Polarization coefficient according to the proposal of Wang and Tsui (2000).

    The Wang-Tsui measure captures polarization by measuring the deviation of
    each group's characteristic from the median, weighted by group size.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to compute the metric.
    K : float, optional
        Scaling constant, by default 1.0.
    gamma : float, optional
        Sensitivity parameter for distance from median, by default 0.5.
        Higher values give more weight to larger deviations.
    value : str, optional
        Column name for population/weight values, by default "value".
    rate : str, optional
        Column name for the characteristic being measured, by default "rate".

    Returns
    -------
    float
        The polarization coefficient.

    References
    ----------
    Wang, Y. Q., & Tsui, K. Y. (2000). Polarization orderings and new classes
    of polarization indices. Journal of Public Economic Theory, 2(3), 349-363.

    See Also
    --------
    esteban_ray : Esteban-Ray polarization measure.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'value': [100, 200], 'rate': [0.4, 0.6]})
    >>> wang_tsui(df)
    0.4472135954999579
    """
    median = np.median(data[rate])
    population = np.sum(data[value])

    result = 0
    for i, row in data.iterrows():
        result += row[value] * np.absolute((row[rate] - median) / median) ** gamma

    return K * result / population
