import pandas as pd


def reynal_querol(
    input_df,
    rate="rate"
):
    return 1 - (input_df[rate].apply(lambda x: (((0.5 - x)/0.5)**2 * x))).sum()
