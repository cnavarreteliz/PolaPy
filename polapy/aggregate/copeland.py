import numpy as np
import pandas as pd
from typing import Union


def copeland(
    data: pd.DataFrame,
    proposal: str = "proposal",
    wins_over: str = "wins_over",
    count: str = "count"
) -> pd.DataFrame:
    """Copeland aggregation from pairwise comparison data.

    Computes Copeland scores for each proposal based on pairwise comparison wins.
    The Copeland score is the number of pairwise wins minus the number of 
    pairwise losses. A proposal wins a pairwise comparison if it has more votes
    than the other proposal in that matchup.

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
        DataFrame with columns [proposal, score] containing Copeland scores.

    References
    ----------
    Copeland, A. H. (1951). A reasonable social welfare function.
    Seminar on Mathematics in Social Sciences.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'proposal': ['A', 'A', 'B', 'B', 'C', 'C'],
    ...     'wins_over': ['B', 'C', 'A', 'C', 'A', 'B'],
    ...     'count': [100, 80, 40, 60, 30, 50]
    ... })
    >>> copeland(df)
    """
    data = data.copy()
    
    # Get all unique proposals
    all_proposals = list(set(data[proposal].unique()) | set(data[wins_over].unique()))
    
    # Create a matrix to track pairwise comparisons
    # Build aggregated vote counts for each pair
    pair_votes = data.groupby([proposal, wins_over])[count].sum().reset_index()
    
    # For each pair (i, j), determine winner
    wins = {p: 0 for p in all_proposals}
    losses = {p: 0 for p in all_proposals}
    
    # Get unique pairs
    pairs_seen = set()
    for _, row in pair_votes.iterrows():
        p1, p2 = row[proposal], row[wins_over]
        pair_key = tuple(sorted([p1, p2]))
        
        if pair_key in pairs_seen:
            continue
        pairs_seen.add(pair_key)
        
        # Get votes for p1 > p2
        votes_p1_over_p2 = pair_votes[
            (pair_votes[proposal] == p1) & (pair_votes[wins_over] == p2)
        ][count].sum()
        
        # Get votes for p2 > p1
        votes_p2_over_p1 = pair_votes[
            (pair_votes[proposal] == p2) & (pair_votes[wins_over] == p1)
        ][count].sum()
        
        if votes_p1_over_p2 > votes_p2_over_p1:
            wins[p1] += 1
            losses[p2] += 1
        elif votes_p2_over_p1 > votes_p1_over_p2:
            wins[p2] += 1
            losses[p1] += 1
        # Ties don't affect the score
    
    # Copeland score = wins - losses
    scores = pd.DataFrame({
        proposal: all_proposals,
        "score": [wins[p] - losses[p] for p in all_proposals]
    })
    
    return scores.sort_values("score", ascending=False).reset_index(drop=True)
