import numpy as np
import pandas as pd
from typing import Union, Tuple


def ahp(
    data: pd.DataFrame,
    proposal: str = "proposal",
    wins_over: str = "wins_over",
    count: str = "count",
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> pd.DataFrame:
    """Analytic Hierarchy Process (AHP) aggregation from pairwise comparison data.

    Computes priority weights using the AHP method based on pairwise comparison
    votes. The method constructs a comparison matrix from vote ratios and 
    computes the principal eigenvector as the priority weights.

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
    max_iterations : int, optional
        Maximum iterations for power method, by default 100.
    tolerance : float, optional
        Convergence tolerance for power method, by default 1e-6.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [proposal, score] containing AHP priority weights
        (normalized to sum to 1).

    References
    ----------
    Saaty, T. L. (1980). The Analytic Hierarchy Process: Planning, Priority
    Setting, Resource Allocation. McGraw-Hill.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'proposal': ['A', 'A', 'B', 'B', 'C', 'C'],
    ...     'wins_over': ['B', 'C', 'A', 'C', 'A', 'B'],
    ...     'count': [100, 80, 40, 60, 30, 50]
    ... })
    >>> ahp(df)
    """
    data = data.copy()
    
    # Get all unique proposals and create index mapping
    all_proposals = sorted(set(data[proposal].unique()) | set(data[wins_over].unique()))
    n = len(all_proposals)
    proposal_to_idx = {p: i for i, p in enumerate(all_proposals)}
    
    # Aggregate votes for each pair
    pair_votes = data.groupby([proposal, wins_over])[count].sum().reset_index()
    
    # Build pairwise comparison matrix
    # A[i,j] = votes(i > j) / votes(j > i) if votes(j > i) > 0, else votes(i > j) + 1
    matrix = np.ones((n, n))
    
    for _, row in pair_votes.iterrows():
        i = proposal_to_idx[row[proposal]]
        j = proposal_to_idx[row[wins_over]]
        votes_i_over_j = row[count]
        matrix[i, j] += votes_i_over_j
    
    # Make matrix reciprocal: A[i,j] = 1 / A[j,i]
    # First, compute ratios
    for i in range(n):
        for j in range(i + 1, n):
            ratio = matrix[i, j] / matrix[j, i] if matrix[j, i] > 0 else matrix[i, j]
            matrix[i, j] = ratio
            matrix[j, i] = 1 / ratio if ratio > 0 else 1
    
    # Diagonal should be 1
    np.fill_diagonal(matrix, 1.0)
    
    # Compute priority vector using power method (principal eigenvector)
    weights = np.ones(n) / n
    
    for _ in range(max_iterations):
        new_weights = matrix @ weights
        new_weights = new_weights / new_weights.sum()
        
        if np.max(np.abs(new_weights - weights)) < tolerance:
            break
        weights = new_weights
    
    scores = pd.DataFrame({
        proposal: all_proposals,
        "score": weights
    })
    
    return scores.sort_values("score", ascending=False).reset_index(drop=True)


def ahp_consistency_ratio(
    data: pd.DataFrame,
    proposal: str = "proposal",
    wins_over: str = "wins_over",
    count: str = "count"
) -> Tuple[float, float]:
    """Calculate the consistency ratio for AHP pairwise comparisons.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with pairwise comparison results.
    proposal : str, optional
        Column name for the winning proposal identifier, by default "proposal".
    wins_over : str, optional
        Column name for the losing proposal identifier, by default "wins_over".
    count : str, optional
        Column name for vote counts, by default "count".

    Returns
    -------
    Tuple[float, float]
        (consistency_index, consistency_ratio)

    References
    ----------
    Saaty, T. L. (1980). The Analytic Hierarchy Process.
    """
    # Random Index values for matrices of size 1-10
    RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 
          6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    
    data = data.copy()
    
    all_proposals = sorted(set(data[proposal].unique()) | set(data[wins_over].unique()))
    n = len(all_proposals)
    
    if n > 10:
        n = 10  # Use RI for n=10 as approximation
    
    if n <= 2:
        return 0.0, 0.0
    
    # Build matrix (same as in ahp function)
    proposal_to_idx = {p: i for i, p in enumerate(all_proposals)}
    pair_votes = data.groupby([proposal, wins_over])[count].sum().reset_index()
    
    matrix = np.ones((len(all_proposals), len(all_proposals)))
    
    for _, row in pair_votes.iterrows():
        i = proposal_to_idx[row[proposal]]
        j = proposal_to_idx[row[wins_over]]
        matrix[i, j] += row[count]
    
    for i in range(len(all_proposals)):
        for j in range(i + 1, len(all_proposals)):
            ratio = matrix[i, j] / matrix[j, i] if matrix[j, i] > 0 else matrix[i, j]
            matrix[i, j] = ratio
            matrix[j, i] = 1 / ratio if ratio > 0 else 1
    
    np.fill_diagonal(matrix, 1.0)
    
    # Compute max eigenvalue
    eigenvalues = np.linalg.eigvals(matrix)
    lambda_max = np.max(np.real(eigenvalues))
    
    # Consistency Index
    CI = (lambda_max - len(all_proposals)) / (len(all_proposals) - 1)
    
    # Consistency Ratio
    ri_value = RI.get(n, 1.49)
    CR = CI / ri_value if ri_value > 0 else 0
    
    return CI, CR
