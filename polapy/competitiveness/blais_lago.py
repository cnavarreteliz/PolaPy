from polapy.aggregate import dhondt
from polapy.stats import quota

import numpy as np
import pandas as pd


def blais_lago(
    input_df,
    candidate="party",
    votes="votes",
    n_seats=1,
    system="dhondt"
):
    if system == "dhondt":
        data, quot = dhondt(input_df, party=candidate,
                            votes=votes, n_seats=n_seats, quota=True)
        seats = dhondt(input_df, party=candidate, votes=votes,
                       n_seats=n_seats, quota=False)

        data["distance"] = data["quota"] - quot
        n_seats = data[data["distance"] >= 0]
        data = data[data["distance"] < 0]

        data["distance"] = data["quota"] - data.apply(
            lambda x: n_seats[n_seats[candidate] != x[candidate]]["quota"].min(), axis=1)
        data["distance"] *= -1

        output_df = pd.merge(
            data.groupby(candidate).agg({"distance": "min"}).reset_index(),
            seats,
            on=candidate,
            how="outer"
        )
        output_df["seats"] = output_df["seats"].fillna(0) + 1
        output_df["seats"] = output_df["distance"] * output_df["seats"]

        output_df = output_df.sort_values("seats", ascending=False)
        output_df = output_df.fillna(0)

    elif system == "hare":
        seats = proportional(input_df, candidate=candidate,
                             votes=votes, method=system, n_seats=n_seats)
        q = quota(method=system,
                  n_votes=input_df[votes].sum(), n_seats=n_seats)
        data = input_df.copy()
        data["quota"] = q

        output_df = pd.merge(
            data,
            seats,
            on=candidate,
            how="outer"
        )

        output_df["seats"] = output_df["seats"].fillna(
            0).apply(lambda x: (2*x + 1)/2 + 0.001)
        output_df["seats"] = q * output_df["seats"] - output_df[votes]

        output_df = output_df.sort_values("seats", ascending=False)
        output_df = output_df.fillna(0)

    elif system == "smp":
        output_df = input_df.copy()
        output_df["seats"] = input_df[votes].max() - output_df[votes]

    output_df["value"] = output_df["seats"] / (input_df[votes].sum()/n_seats)

    return output_df["value"].sum(), output_df
