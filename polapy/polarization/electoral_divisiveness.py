import numpy as np
import pandas as pd


def electoral_divisiveness(
    data: pd.DataFrame,
    unit: str = "unit",
    candidate: str = "candidate",
    votes: str = "votes",
    score: str = "score"
) -> tuple:
    """Electoral divisiveness coefficient according to Navarrete et al. (2023).

    The electoral divisiveness measure captures the dispersion level of the 
    performance of candidates across electoral units. It measures how much
    a candidate's support varies geographically.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to compute the metric. Must contain columns for electoral units,
        candidates, and votes.
    unit : str, optional
        Column name for electoral unit identifiers (e.g., districts, regions),
        by default "unit".
    candidate : str, optional
        Column name for candidate/party identifiers, by default "candidate".
    votes : str, optional
        Column name for vote counts, by default "votes".
    score : str, optional
        Column name for vote share. If not present in data, it will be
        computed as the proportion of votes within each unit, by default "score".

    Returns
    -------
    tuple
        (float, pd.DataFrame): 
        - float: The electoral divisiveness coefficient (sum of antagonism values).
        - pd.DataFrame: DataFrame with antagonism values for each candidate.
          Columns: [candidate, weight, antagonism]

    References
    ----------
    Navarrete, C., et al. (2023). Electoral Divisiveness: A New Measure for
    Candidate Performance Dispersion.

    See Also
    --------
    esteban_ray : Esteban-Ray polarization measure.
    election_competitiveness : Election competitiveness measure.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'unit': ['A', 'A', 'B', 'B'],
    ...     'candidate': ['X', 'Y', 'X', 'Y'],
    ...     'votes': [100, 50, 60, 90]
    ... })
    >>> value, details = electoral_divisiveness(df)
    """
    data = data.copy()

    values = data.pivot(
        index=candidate,
        columns=unit,
        values=votes
    )

    if score not in list(data):
        data[score] = data.groupby(unit, group_keys=False)[votes]\
            .apply(lambda x: x / x.sum())

    # Gets total of votes and candidates
    N_candidates = len(data[candidate].unique())
    total = values.sum().sum()

    def get_average(x, total=1):
        return x[votes].sum() / total

    df_mean = data.groupby(candidate)\
        .apply(lambda x: get_average(x, total=total))\
        .reset_index().rename(columns={0: "weight"})

    xx = np.sum(values).reset_index().rename(columns={0: "total"})

    df_within = pd.merge(data, df_mean, on=candidate)
    df_within = pd.merge(df_within, xx, on=unit)

    df_within["diff_abs"] = np.absolute(df_within[score] - df_within["weight"])
    df_within["total"] = df_within[votes] * df_within["diff_abs"]

    df_within = df_within.groupby(candidate).agg(
        {"total": "sum"}) / (N_candidates - 1)

    df_sum = data.groupby(candidate).agg(
        {votes: "sum"}).reset_index().rename(columns={votes: "total_votes"})

    df_within = pd.merge(df_within, df_sum, on=candidate)
    df_within = df_within.rename(
        columns={"total": votes, votes: "weight"}
    )
    df_within[votes] = df_within.apply(
        lambda x: x[votes] / x["total_votes"] if x["total_votes"] > 0 else 0, axis=1)
    df_within = df_within.drop(columns=["total_votes"])

    value = df_within[votes].sum()

    return value, df_within.rename(columns={votes: "antagonism"})


# Backward compatibility alias
election_polarization = electoral_divisiveness
