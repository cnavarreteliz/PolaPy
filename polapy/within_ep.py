import numpy as np
import pandas as pd


def within_ep(
    data: pd.DataFrame,
    unit: str = "unit",
    candidate: str = "candidate",
    votes: str = "votes",
    score: str = "score"
) -> (float, pd.DataFrame):
    """Within-EP coefficient according to the proposal of Navarrete et al. (2023)

    The Within-EP captures the dispersion level of the performance of candidates.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to compute the metric.
    unit, candidate, votes, score : names of variables in data, optional
        Inputs for interpreting the DataFrame. If no values are specified, they are interpreted as unit, candidate, votes, and score respectively. If score column is not found in data, the algorithm automatically computes a new score column.

    Returns
    -------
    float
        Within-EP
    pd.DataFrame
        _description_
    """
    data = data.copy()

    values = data.pivot(
        index=candidate,
        columns=unit,
        values=votes
    )

    if score not in list(data):
        data[score] = data.groupby(unit, group_keys=False)[votes]\
            .apply(lambda x: x/x.sum())

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
