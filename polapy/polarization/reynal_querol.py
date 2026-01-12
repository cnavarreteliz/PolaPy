import pandas as pd


def reynal_querol(
    data: pd.DataFrame,
    rate: str = "rate"
) -> float:
    """Polarization coefficient according to Reynal-Querol (2002).

    This measure is designed particularly for ethnic and religious fractionalization,
    capturing how far the distribution is from a bipolar distribution (two groups
    of equal size). Maximum polarization occurs when there are exactly two groups
    of equal size (0.5 each).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to compute the metric. Must contain a column with group shares.
    rate : str, optional
        Column name for the share/proportion of each group (should sum to 1),
        by default "rate".

    Returns
    -------
    float
        The polarization coefficient, ranging from 0 (no polarization) to 1
        (maximum bipolar polarization).

    References
    ----------
    Reynal-Querol, M. (2002). Ethnicity, political systems, and civil wars.
    Journal of Conflict Resolution, 46(1), 29-54.

    See Also
    --------
    esteban_ray : Esteban-Ray polarization measure.

    Notes
    -----
    The Reynal-Querol index is calculated as:

    .. math::
        RQ = 1 - \\sum_{i=1}^{N} \\left(\\frac{0.5 - \\pi_i}{0.5}\\right)^2 \\pi_i

    where :math:`\\pi_i` is the population share of group :math:`i`.

    Examples
    --------
    >>> import pandas as pd
    >>> # Two equal groups: maximum polarization
    >>> df = pd.DataFrame({'rate': [0.5, 0.5]})
    >>> reynal_querol(df)
    1.0

    >>> # Four groups: moderate polarization
    >>> df = pd.DataFrame({'rate': [0.4, 0.3, 0.2, 0.1]})
    >>> reynal_querol(df)
    0.8
    """
    return 1 - (data[rate].apply(lambda x: (((0.5 - x) / 0.5) ** 2 * x))).sum()
