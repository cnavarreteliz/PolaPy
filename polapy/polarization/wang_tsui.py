import numpy as np
import pandas as pd


def wang_tsui(
    input_df,
    K: int = 1,
    gamma: float = 0.5,
    value="value",
    rate="rate"
):
    median = np.median(input_df["rate"])
    population = np.sum(input_df["value"])

    value = 0
    for i, xx in input_df.iterrows():
        value += xx.value * np.absolute((xx.rate - median)/median) ** gamma

    return K * value / population
