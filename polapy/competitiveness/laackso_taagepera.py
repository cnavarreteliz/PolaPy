import pandas as pd


def laackso_taagepera(
    data: pd.DataFrame,
    share: str = "share",
    alpha: float = 2.0
) -> float:
    """Effective Number of Parties according to Laakso and Taagepera (1979).

    Calculates the "effective" number of parties, which weights parties by their
    relative strength. A system with two equally-sized parties has an effective
    number of 2.0, while a system with one dominant party approaches 1.0.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with party vote/seat shares.
    share : str, optional
        Column name for the party share (vote share or seat share),
        should sum to 1, by default "share".
    alpha : float, optional
        The exponent parameter. When alpha=2, this gives the standard
        Laakso-Taagepera index. Setting alpha=1 gives the exponential
        of Shannon entropy, by default 2.0.

    Returns
    -------
    float
        The effective number of parties.

    References
    ----------
    Laakso, M., & Taagepera, R. (1979). "Effective" number of parties:
    A measure with application to West Europe. Comparative Political Studies,
    12(1), 3-27.

    Notes
    -----
    The Laakso-Taagepera index is calculated as:

    .. math::
        N_{eff} = \\left(\\sum_{i=1}^{N} p_i^{\\alpha}\\right)^{\\frac{1}{1-\\alpha}}

    where :math:`p_i` is the vote or seat share of party :math:`i`.

    For the standard case (alpha=2):

    .. math::
        N_{eff} = \\frac{1}{\\sum_{i=1}^{N} p_i^2}

    Examples
    --------
    >>> import pandas as pd
    >>> # Two equal parties
    >>> df = pd.DataFrame({'share': [0.5, 0.5]})
    >>> laackso_taagepera(df)
    2.0

    >>> # One dominant party
    >>> df = pd.DataFrame({'share': [0.9, 0.05, 0.05]})
    >>> round(laackso_taagepera(df), 2)
    1.22
    """
    return (data[share].apply(lambda x: x ** alpha).sum()) ** (1 / (1 - alpha))
