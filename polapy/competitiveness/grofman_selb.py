from polapy.aggregate import dhondt
import pandas as pd


def grofman_selb(
    input_df,
    n_seats: int = 5,
    candidate: str = "party",
    votes: str = "Votes"
) -> (float, pd.DataFrame):

    T_e = 1/(n_seats + 1)  # Threshold of Exclusion

    data, quot = dhondt(input_df, party=candidate,
                        votes=votes, n_seats=n_seats, quota=True)
    seats = dhondt(input_df, party=candidate, votes=votes,
                   n_seats=n_seats, quota=False)
    data["distance"] = data["quota"] - quot

    frag = data[data["distance"] < 0]
    data = data[data["distance"] >= 0]

    lose_seat = pd.merge(
        seats,
        frag.groupby(candidate).agg({"quota": "max"}).reset_index(),
        on=candidate
    )
    lose_seat = pd.merge(lose_seat, input_df, on=candidate)

    def get_loss(x):
        frag = lose_seat[lose_seat[candidate] != x[candidate]]
        curr = lose_seat[lose_seat[candidate] == x[candidate]]
        s = list(curr["seats"])[0]
        v = list(curr[votes])[0]

        frag["den"] = (frag["seats"] + 1) + s
        frag["num"] = (frag["seats"] + 1) * v - s * lose_seat[votes]
        frag["value"] = frag["num"] / frag["den"]

        return frag["value"].min()

    input_lose = data.groupby(candidate).agg({"quota": "min"}).reset_index()
    input_lose["lose"] = input_lose.apply(lambda x: get_loss(x), axis=1)

    input_gain = pd.merge(input_df, seats, on=candidate, how="outer").fillna(0)
    input_gain["gain"] = input_gain.apply(lambda x: (
        (1 + x["seats"])/(n_seats + 1)) - x[votes], axis=1)

    output = pd.merge(input_gain, input_lose, on=candidate)
    output["competition"] = output.apply(lambda x: max(
        (T_e - x["gain"]), (T_e - x["lose"])), axis=1) / T_e

    # Index of Competition, C
    C = (output["competition"] * output[votes]).sum()

    return C, output
