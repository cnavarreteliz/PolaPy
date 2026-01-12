from polapy.aggregate import dhondt
import pandas as pd


def grofman_selb(
    data: pd.DataFrame,
    n_seats: int = 5,
    candidate: str = "party",
    votes: str = "votes"
) -> tuple:
    """Competition index according to Grofman and Selb (2009).

    Measures electoral competition by calculating how close parties are to
    gaining or losing seats. Based on the threshold of exclusion and the
    D'Hondt seat allocation method.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with party/candidate votes.
    n_seats : int, optional
        Number of seats in the district, by default 5.
    candidate : str, optional
        Column name for party/candidate identifiers, by default "party".
    votes : str, optional
        Column name for vote counts, by default "votes".

    Returns
    -------
    tuple
        (float, pd.DataFrame):
        - float: The competition index C (weighted by vote share).
        - pd.DataFrame: Detailed results with columns for each party's
          gain/loss margins and competition scores.

    References
    ----------
    Grofman, B., & Selb, P. (2009). A fully general index of political
    competition. Electoral Studies, 28(2), 291-296.

    See Also
    --------
    blais_lago : Blais-Lago competition measure.
    dhondt : D'Hondt seat allocation method.

    Notes
    -----
    The threshold of exclusion :math:`T_e = 1/(n+1)` where :math:`n` is the
    number of seats. A party needs more than this vote share to guarantee
    winning at least one seat.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'party': ['A', 'B', 'C'], 'votes': [5000, 3000, 2000]})
    >>> value, details = grofman_selb(df, n_seats=5)
    """
    T_e = 1 / (n_seats + 1)  # Threshold of Exclusion

    dhondt_data, quot = dhondt(data, party=candidate,
                               votes=votes, n_seats=n_seats, quota=True)
    seats = dhondt(data, party=candidate, votes=votes,
                   n_seats=n_seats, quota=False)
    dhondt_data["distance"] = dhondt_data["quota"] - quot

    frag = dhondt_data[dhondt_data["distance"] < 0]
    dhondt_data = dhondt_data[dhondt_data["distance"] >= 0]

    lose_seat = pd.merge(
        seats,
        frag.groupby(candidate).agg({"quota": "max"}).reset_index(),
        on=candidate
    )
    lose_seat = pd.merge(lose_seat, data, on=candidate)

    def get_loss(x):
        frag_df = lose_seat[lose_seat[candidate] != x[candidate]]
        curr = lose_seat[lose_seat[candidate] == x[candidate]]
        s = list(curr["seats"])[0]
        v = list(curr[votes])[0]

        frag_df = frag_df.copy()
        frag_df["den"] = (frag_df["seats"] + 1) + s
        frag_df["num"] = (frag_df["seats"] + 1) * v - s * lose_seat[votes]
        frag_df["value"] = frag_df["num"] / frag_df["den"]

        return frag_df["value"].min()

    input_lose = dhondt_data.groupby(candidate).agg({"quota": "min"}).reset_index()
    input_lose["lose"] = input_lose.apply(lambda x: get_loss(x), axis=1)

    input_gain = pd.merge(data, seats, on=candidate, how="outer").fillna(0)
    input_gain["gain"] = input_gain.apply(lambda x: (
        (1 + x["seats"]) / (n_seats + 1)) - x[votes], axis=1)

    output = pd.merge(input_gain, input_lose, on=candidate)
    output["competition"] = output.apply(lambda x: max(
        (T_e - x["gain"]), (T_e - x["lose"])), axis=1) / T_e

    # Index of Competition, C
    C = (output["competition"] * output[votes]).sum()

    return C, output
