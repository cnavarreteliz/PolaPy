import numpy as np
import pandas as pd
from typing import Union


def winrate(
    data: pd.DataFrame,
    proposal: str = "proposal",
    wins_over: str = "wins_over",
    count: str = "count"
) -> pd.DataFrame:
    """Win rate aggregation from pairwise comparison data.

    Computes the win rate for each proposal based on pairwise comparisons.
    The win rate is calculated as the number of votes received divided by
    the total votes in all matchups involving that proposal.

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
        DataFrame with columns [proposal, score] containing win rates (0 to 1).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'proposal': ['A', 'A', 'B', 'B', 'C', 'C'],
    ...     'wins_over': ['B', 'C', 'A', 'C', 'A', 'B'],
    ...     'count': [100, 80, 40, 60, 30, 50]
    ... })
    >>> winrate(df)
    """
    data = data.copy()
    
    # Get all unique proposals
    all_proposals = list(set(data[proposal].unique()) | set(data[wins_over].unique()))
    
    # Aggregate votes for each pair
    pair_votes = data.groupby([proposal, wins_over])[count].sum().reset_index()
    
    # Calculate wins and total matchups for each proposal
    wins = {p: 0 for p in all_proposals}
    total_votes = {p: 0 for p in all_proposals}
    
    for _, row in pair_votes.iterrows():
        winner = row[proposal]
        loser = row[wins_over]
        votes = row[count]
        
        wins[winner] += votes
        total_votes[winner] += votes
        total_votes[loser] += votes
    
    # Calculate win rate
    scores = pd.DataFrame({
        proposal: all_proposals,
        "score": [wins[p] / total_votes[p] if total_votes[p] > 0 else 0 
                  for p in all_proposals]
    })
    
    return scores.sort_values("score", ascending=False).reset_index(drop=True)
