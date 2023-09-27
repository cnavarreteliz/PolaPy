import numpy as np
import pandas as pd


def between_ep(
    data,
    unit: str = "unit",
    candidate: str = "candidate",
    votes: str = "votes",
    score: str = "score"
) -> (float, pd.DataFrame):
    data = data.copy()

    if score not in list(data):
        data[score] = data.groupby(unit, group_keys=False)[votes]\
            .apply(lambda x: x/x.sum())

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

    for candidate in rates.index:
        N_candidates = rates.shape[0]
        rates_c = rates[rates.index == candidate]
        values_c = values[values.index == candidate]

        xx = np.tile(rates_c, reps=(N_candidates, 1))
        yy = np.tile(values_c, reps=(N_candidates, 1))

        between = np.multiply(yy, (1 - np.absolute(xx - rates)))
        between = between[between.index != candidate]

        dv_between = 0 if np.sum(values_c).sum() == 0 else np.sum(
            between).sum() / (N_candidates * (N_candidates - 1) * np.sum(values_c).sum())

        output.append({
            "candidate": candidate,
            "antagonism": dv_between
        })

    df_between = pd.DataFrame(output)
    value = df_between["antagonism"].sum()

    return value, df_between
