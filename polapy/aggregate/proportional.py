import pandas as pd


def proportional(
    data: pd.DataFrame,
    candidate: str = "party",
    votes: str = "votes",
    method: str = "hare",
    n_seats: int = 1
) -> pd.DataFrame:
    """Proportional seat allocation using Largest Remainder Method.

    Allocates seats proportionally to parties based on their vote share,
    using the specified quota method.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with party votes.
    candidate : str, optional
        Column name for party/candidate identifiers, by default "party".
    votes : str, optional
        Column name for vote counts, by default "votes".
    method : str, optional
        Quota method: "hare", "droop", or "imperiali", by default "hare".
    n_seats : int, optional
        Number of seats to allocate, by default 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [candidate, seats] showing seat allocation.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'party': ['A', 'B', 'C'], 'votes': [50000, 30000, 20000]})
    >>> proportional(df, n_seats=10)
    """
    from polapy.stats import quota as quota_fn

    total_votes = data[votes].sum()
    q = quota_fn(method=method, n_votes=total_votes, n_seats=n_seats)

    result = data[[candidate, votes]].copy()
    result["quota"] = result[votes] / q
    result["seats"] = result["quota"].apply(lambda x: int(x))
    result["remainder"] = result["quota"] - result["seats"]

    # Allocate remaining seats by largest remainder
    allocated = result["seats"].sum()
    remaining = n_seats - allocated

    if remaining > 0:
        result = result.sort_values("remainder", ascending=False)
        for i, idx in enumerate(result.index):
            if i < remaining:
                result.loc[idx, "seats"] += 1

    return result[[candidate, "seats"]].sort_values("seats", ascending=False).reset_index(drop=True)
