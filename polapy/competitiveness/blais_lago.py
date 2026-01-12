from polapy.aggregate import dhondt
from polapy.aggregate.proportional import proportional
from polapy.stats import quota

import numpy as np
import pandas as pd


def blais_lago(
    data: pd.DataFrame,
    candidate: str = "party",
    votes: str = "votes",
    n_seats: int = 1,
    system: str = "dhondt"
) -> tuple:
    """Election competitiveness according to Blais and Lago (2009).

    Measures how competitive an election is by calculating the minimum
    vote change needed for each party to gain or lose a seat.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with party/candidate votes.
    candidate : str, optional
        Column name for party/candidate identifiers, by default "party".
    votes : str, optional
        Column name for vote counts, by default "votes".
    n_seats : int, optional
        Number of seats in the election, by default 1.
    system : str, optional
        Electoral system: "dhondt", "hare", or "smp", by default "dhondt".

    Returns
    -------
    tuple
        (float, pd.DataFrame): Competition index and detailed results per party.

    References
    ----------
    Blais, A., & Lago, I. (2009). A general measure of district competitiveness.
    Electoral Studies, 28(1), 94-100.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'party': ['A', 'B', 'C'], 'votes': [5000, 3000, 2000]})
    >>> value, details = blais_lago(df, n_seats=5)
    """
    data = data.copy()

    if system == "dhondt":
        dhondt_data, quot = dhondt(data, party=candidate,
                                   votes=votes, n_seats=n_seats, quota=True)
        seats = dhondt(data, party=candidate, votes=votes,
                       n_seats=n_seats, quota=False)

        dhondt_data["distance"] = dhondt_data["quota"] - quot
        n_seats_df = dhondt_data[dhondt_data["distance"] >= 0]
        dhondt_data = dhondt_data[dhondt_data["distance"] < 0]

        dhondt_data["distance"] = dhondt_data["quota"] - dhondt_data.apply(
            lambda x: n_seats_df[n_seats_df[candidate] != x[candidate]]["quota"].min(), axis=1)
        dhondt_data["distance"] *= -1

        output_df = pd.merge(
            dhondt_data.groupby(candidate).agg({"distance": "min"}).reset_index(),
            seats,
            on=candidate,
            how="outer"
        )
        output_df["seats"] = output_df["seats"].fillna(0) + 1
        output_df["seats"] = output_df["distance"] * output_df["seats"]

        output_df = output_df.sort_values("seats", ascending=False)
        output_df = output_df.fillna(0)

    elif system == "hare":
        seats = proportional(data, candidate=candidate,
                             votes=votes, method="hare", n_seats=n_seats)
        q = quota(method="hare",
                  n_votes=data[votes].sum(), n_seats=n_seats)
        data["quota"] = q

        output_df = pd.merge(
            data,
            seats,
            on=candidate,
            how="outer"
        )

        output_df["seats"] = output_df["seats"].fillna(
            0).apply(lambda x: (2 * x + 1) / 2 + 0.001)
        output_df["seats"] = q * output_df["seats"] - output_df[votes]

        output_df = output_df.sort_values("seats", ascending=False)
        output_df = output_df.fillna(0)

    elif system == "smp":
        output_df = data.copy()
        output_df["seats"] = data[votes].max() - output_df[votes]

    output_df["value"] = output_df["seats"] / (data[votes].sum() / n_seats)

    return output_df["value"].sum(), output_df
