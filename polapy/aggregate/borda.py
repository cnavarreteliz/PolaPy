import numpy as np
import pandas as pd
from typing import Union


def borda(
    data: pd.DataFrame,
    proposal: str = "proposal",
    wins_over: str = "wins_over",
    count: str = "count"
) -> pd.DataFrame:
    """Borda count aggregation from pairwise comparison data.

    Computes Borda scores for each proposal based on pairwise comparison votes.
    The Borda score for a proposal is the sum of votes it receives against each
    other proposal.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with pairwise comparison results. Expected columns are:
        - proposal: identifier for the winning proposal in each comparison
        - wins_over: identifier for the losing proposal in each comparison  
        - count: number of times this preference was expressed
    proposal : str, optional
        Column name for the winning proposal identifier, by default "proposal".
    wins_over : str, optional
        Column name for the losing proposal identifier, by default "wins_over".
    count : str, optional
        Column name for vote counts, by default "count".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [proposal, score] containing Borda scores.

    References
    ----------
    Borda, J. C. (1781). Mémoire sur les élections au scrutin.
    Histoire de l'Académie Royale des Sciences.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'proposal': ['A', 'A', 'B'],
    ...     'wins_over': ['B', 'C', 'C'],
    ...     'count': [100, 80, 60]
    ... })
    >>> borda(df)
    """
    data = data.copy()
    
    # Sum votes for each proposal (number of wins)
    scores = data.groupby(proposal)[count].sum().reset_index()
    scores.columns = [proposal, "score"]
    
    # Get all unique proposals (including those that only appear as losers)
    all_proposals = set(data[proposal].unique()) | set(data[wins_over].unique())
    
    # Ensure all proposals are in the result
    all_proposals_df = pd.DataFrame({proposal: list(all_proposals)})
    scores = pd.merge(all_proposals_df, scores, on=proposal, how="left")
    scores["score"] = scores["score"].fillna(0)
    
    return scores.sort_values("score", ascending=False).reset_index(drop=True)
