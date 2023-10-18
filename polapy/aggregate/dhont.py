import pandas as pd


def dhondt(
    input_df,
    party: str = "party",
    votes: str = "votes",
    n_seats: int = 1,
    quota: bool = False
) -> pd.DataFrame:
    """D'Hondt (or Jefferson) method.

    Calculates the number of elected alternatives of each party using the D'Hondt (or Jefferson) method.

    Parameters
    ----------
    n_seats : int, default=1, optional:
        Number of seats to be assigned in the election.

    Returns
    -------
    pandas.DataFrame:
        Summary with the seats of each party.
    """
    output = []
    for __party__, df_tmp in input_df.groupby(party):
        __votes__ = df_tmp[votes].values[0]
        for i in range(n_seats):
            output.append({
                party: __party__,
                "quota": __votes__ / (i + 1)
            })

    tmp = pd.DataFrame(output).sort_values(
        "quota", ascending=False).reset_index(drop=True)

    if quota:
        return tmp, tmp.loc[n_seats - 1, "quota"]

    return tmp.head(n_seats).groupby(party).count().reset_index().rename(columns={"quota": "seats"}).sort_values("seats", ascending=False)
