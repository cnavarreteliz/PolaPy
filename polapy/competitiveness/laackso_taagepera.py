def laackso_taagepera(
    input_df,
    share="share",
    alpha=2
):
    return (input_df[share].apply(lambda x: x**alpha).sum())**(1/(1 - alpha))
