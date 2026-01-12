import numpy as np
import pandas as pd


def election_competitiveness(
    data: pd.DataFrame,
    unit: str = "unit",
    candidate: str = "candidate",
    votes: str = "votes",
    score: str = "score"
) -> tuple:
    """Election competitiveness measure based on between-candidate interactions.

    Calculates the competitiveness of an election by measuring how closely
    matched candidates are across electoral units. Higher values indicate
    more competitive elections.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with electoral results. Must contain columns for units,
        candidates, and votes.
    unit : str, optional
        Column name for electoral unit identifiers, by default "unit".
    candidate : str, optional
        Column name for candidate/party identifiers, by default "candidate".
    votes : str, optional
        Column name for vote counts, by default "votes".
    score : str, optional
        Column name for vote share. If not present, it will be computed
        as the proportion of votes within each unit, by default "score".

    Returns
    -------
    tuple
        (float, pd.DataFrame):
        - float: The overall competitiveness index.
        - pd.DataFrame: DataFrame with competitiveness values for each candidate.

    See Also
    --------
    electoral_divisiveness : Electoral divisiveness measure.
    blais_lago : Blais-Lago competition measure.
    grofman_selb : Grofman-Selb competition measure.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'unit': ['A', 'A', 'B', 'B'],
    ...     'candidate': ['X', 'Y', 'X', 'Y'],
    ...     'votes': [55, 45, 48, 52]
    ... })
    >>> value, details = election_competitiveness(df)
    """
    data = data.copy()

    if score not in list(data):
        data[score] = data.groupby(unit, group_keys=False)[votes]\
            .apply(lambda x: x / x.sum())

    values = data.pivot(
        index=candidate,
        columns=unit,
        values=votes
    )
    rates = data.pivot(
        index=candidate,
        columns=unit,
        values=score
    )

    output = []

    for cand in rates.index:
        N_candidates = rates.shape[0]
        rates_c = rates[rates.index == cand]
        values_c = values[values.index == cand]

        xx = np.tile(rates_c, reps=(N_candidates, 1))
        yy = np.tile(values_c, reps=(N_candidates, 1))

        between = np.multiply(yy, (1 - np.absolute(xx - rates)))
        between = between[between.index != cand]

        dv_between = 0 if np.sum(values_c).sum() == 0 else np.sum(
            between).sum() / (N_candidates * (N_candidates - 1) * np.sum(values_c).sum())

        output.append({
            "candidate": cand,
            "antagonism": dv_between
        })

    df_between = pd.DataFrame(output)
    value = df_between["antagonism"].sum()

    return value, df_between
