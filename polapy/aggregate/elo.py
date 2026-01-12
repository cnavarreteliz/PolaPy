import numpy as np
import pandas as pd
from typing import Union


def elo(
    data: pd.DataFrame,
    proposal: str = "proposal",
    wins_over: str = "wins_over",
    count: str = "count",
    base_rating: float = 1000.0,
    k_factor: float = 32.0,
    iterations: int = 1
) -> pd.DataFrame:
    """Elo rating aggregation from pairwise comparison data.

    Computes Elo ratings for each proposal based on pairwise comparison results.
    The Elo system updates ratings based on expected vs actual outcomes of
    pairwise matchups.

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
    base_rating : float, optional
        Initial rating for all proposals, by default 1000.0.
    k_factor : float, optional
        Maximum rating change per game (K-factor), by default 32.0.
    iterations : int, optional
        Number of times to iterate through all matchups, by default 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [proposal, score] containing Elo ratings.

    References
    ----------
    Elo, A. E. (1978). The Rating of Chessplayers, Past and Present.
    Arco Publishing.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'proposal': ['A', 'A', 'B'],
    ...     'wins_over': ['B', 'C', 'C'],
    ...     'count': [100, 80, 60]
    ... })
    >>> elo(df)
    """
    data = data.copy()
    
    # Get all unique proposals
    all_proposals = list(set(data[proposal].unique()) | set(data[wins_over].unique()))
    
    # Initialize ratings
    ratings = {p: base_rating for p in all_proposals}
    
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    # Process each matchup
    for _ in range(iterations):
        for _, row in data.iterrows():
            winner = row[proposal]
            loser = row[wins_over]
            match_count = row[count]
            
            # Calculate expected scores
            exp_winner = expected_score(ratings[winner], ratings[loser])
            exp_loser = expected_score(ratings[loser], ratings[winner])
            
            # Update ratings (winner gets 1, loser gets 0)
            # Scale K by the number of matches in this comparison
            k_scaled = k_factor * np.log1p(match_count)
            
            ratings[winner] += k_scaled * (1 - exp_winner)
            ratings[loser] += k_scaled * (0 - exp_loser)
    
    scores = pd.DataFrame({
        proposal: list(ratings.keys()),
        "score": list(ratings.values())
    })
    
    return scores.sort_values("score", ascending=False).reset_index(drop=True)
