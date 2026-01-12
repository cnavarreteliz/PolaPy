import numpy as np
import pandas as pd
from typing import Union, Callable, Literal

from .borda import borda
from .copeland import copeland
from .ahp import ahp
from .elo import elo
from .winrate import winrate


# Available aggregation methods
AGGREGATION_METHODS = {
    "borda": borda,
    "copeland": copeland,
    "ahp": ahp,
    "elo": elo,
    "winrate": winrate
}


def divisiveness(
    data: pd.DataFrame,
    proposal: str = "proposal",
    wins_over: str = "wins_over",
    count: str = "count",
    voter: str = "voter",
    method: Union[str, Callable] = "borda"
) -> tuple:
    """Divisiveness measure for pairwise comparison data.

    Computes the divisiveness of each proposal based on how differently
    subpopulations evaluate it. For each proposal P_i, we consider the 
    subpopulation that preferred P_i over P_j and compute the score difference
    between these two subpopulations.

    The divisiveness D_i is defined as:
    
        D_i = (1 / (N-1)) * Σ_{j≠i} √((S_i(P_i > P_j) - S_i(P_j > P_i))²)
    
    Where:
    - S_i(P_i > P_j) is the score of P_i computed only from voters who 
      preferred P_i over P_j
    - S_i(P_j > P_i) is the score of P_i computed only from voters who 
      preferred P_j over P_i
    - N is the total number of proposals

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with individual pairwise comparison votes. Expected columns are:
        - voter: identifier for the voter
        - proposal: identifier for the winning proposal in each comparison
        - wins_over: identifier for the losing proposal in each comparison  
        - count: number of times this preference was expressed (usually 1)
    proposal : str, optional
        Column name for the winning proposal identifier, by default "proposal".
    wins_over : str, optional
        Column name for the losing proposal identifier, by default "wins_over".
    count : str, optional
        Column name for vote counts, by default "count".
    voter : str, optional
        Column name for voter identifier, by default "voter".
    method : Union[str, Callable], optional
        Aggregation method to use. Can be one of: "borda", "copeland", "ahp", 
        "elo", "winrate", or a custom function. By default "borda".

    Returns
    -------
    tuple
        (float, pd.DataFrame): 
        - float: The mean divisiveness across all proposals.
        - pd.DataFrame: DataFrame with divisiveness values for each proposal.
          Columns: [proposal, divisiveness]

    References
    ----------
    Based on measures of polarization in the literature. This definition
    resembles standard deviation and can be interpreted as a second 
    statistical moment for any aggregation function S.

    See Also
    --------
    borda : Borda count aggregation.
    copeland : Copeland aggregation.
    ahp : Analytic Hierarchy Process aggregation.
    elo : Elo rating aggregation.
    winrate : Win rate aggregation.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'voter': ['V1', 'V1', 'V2', 'V2', 'V3', 'V3'],
    ...     'proposal': ['A', 'A', 'B', 'B', 'A', 'B'],
    ...     'wins_over': ['B', 'C', 'A', 'C', 'C', 'C'],
    ...     'count': [1, 1, 1, 1, 1, 1]
    ... })
    >>> value, details = divisiveness(df, method="borda")
    """
    data = data.copy()
    
    # Get the aggregation function
    if isinstance(method, str):
        if method not in AGGREGATION_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Available methods: "
                f"{list(AGGREGATION_METHODS.keys())}"
            )
        agg_func = AGGREGATION_METHODS[method]
    else:
        agg_func = method
    
    # Get all unique proposals
    all_proposals = sorted(
        set(data[proposal].unique()) | set(data[wins_over].unique())
    )
    N = len(all_proposals)
    
    if N < 2:
        return 0.0, pd.DataFrame({proposal: all_proposals, "divisiveness": [0.0] * N})
    
    # For each pair of proposals (P_i, P_j), identify voters who preferred
    # P_i over P_j and those who preferred P_j over P_i
    divisiveness_values = {}
    
    for prop_i in all_proposals:
        squared_diffs_sum = 0.0
        n_pairs = 0
        
        for prop_j in all_proposals:
            if prop_i == prop_j:
                continue
            
            # Voters who preferred P_i over P_j
            voters_i_over_j = data[
                (data[proposal] == prop_i) & (data[wins_over] == prop_j)
            ][voter].unique()
            
            # Voters who preferred P_j over P_i
            voters_j_over_i = data[
                (data[proposal] == prop_j) & (data[wins_over] == prop_i)
            ][voter].unique()
            
            # Compute score of P_i among voters who preferred P_i over P_j
            if len(voters_i_over_j) > 0:
                data_subset_i = data[data[voter].isin(voters_i_over_j)]
                # Aggregate data from this subpopulation
                data_agg_i = data_subset_i.groupby([proposal, wins_over])[count].sum().reset_index()
                scores_i = agg_func(
                    data_agg_i, 
                    proposal=proposal, 
                    wins_over=wins_over, 
                    count=count
                )
                score_i_from_i_voters = scores_i[
                    scores_i[proposal] == prop_i
                ]["score"].values
                score_i_from_i_voters = score_i_from_i_voters[0] if len(score_i_from_i_voters) > 0 else 0.0
            else:
                score_i_from_i_voters = 0.0
            
            # Compute score of P_i among voters who preferred P_j over P_i
            if len(voters_j_over_i) > 0:
                data_subset_j = data[data[voter].isin(voters_j_over_i)]
                # Aggregate data from this subpopulation
                data_agg_j = data_subset_j.groupby([proposal, wins_over])[count].sum().reset_index()
                scores_j = agg_func(
                    data_agg_j, 
                    proposal=proposal, 
                    wins_over=wins_over, 
                    count=count
                )
                score_i_from_j_voters = scores_j[
                    scores_j[proposal] == prop_i
                ]["score"].values
                score_i_from_j_voters = score_i_from_j_voters[0] if len(score_i_from_j_voters) > 0 else 0.0
            else:
                score_i_from_j_voters = 0.0
            
            # Compute squared difference
            diff = score_i_from_i_voters - score_i_from_j_voters
            squared_diffs_sum += diff ** 2
            n_pairs += 1
        
        # Divisiveness for proposal i
        if n_pairs > 0:
            divisiveness_values[prop_i] = np.sqrt(squared_diffs_sum) / (N - 1)
        else:
            divisiveness_values[prop_i] = 0.0
    
    # Create output DataFrame
    result_df = pd.DataFrame({
        proposal: list(divisiveness_values.keys()),
        "divisiveness": list(divisiveness_values.values())
    })
    result_df = result_df.sort_values("divisiveness", ascending=False).reset_index(drop=True)
    
    # Mean divisiveness
    mean_divisiveness = result_df["divisiveness"].mean()
    
    return mean_divisiveness, result_df


def divisiveness_simple(
    data: pd.DataFrame,
    proposal: str = "proposal",
    wins_over: str = "wins_over",
    count: str = "count",
    method: Union[str, Callable] = "borda"
) -> tuple:
    """Simplified divisiveness measure without individual voter tracking.

    For datasets where individual voter data is not available, this function
    computes divisiveness using only the pairwise comparison counts. It 
    estimates the score difference by computing scores from each directed
    comparison separately.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with aggregated pairwise comparison results.
    proposal : str, optional
        Column name for the winning proposal identifier, by default "proposal".
    wins_over : str, optional
        Column name for the losing proposal identifier, by default "wins_over".
    count : str, optional
        Column name for vote counts, by default "count".
    method : Union[str, Callable], optional
        Aggregation method to use, by default "borda".

    Returns
    -------
    tuple
        (float, pd.DataFrame): Mean divisiveness and per-proposal values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'proposal': ['A', 'A', 'B', 'B', 'C', 'C'],
    ...     'wins_over': ['B', 'C', 'A', 'C', 'A', 'B'],
    ...     'count': [100, 80, 40, 60, 30, 50]
    ... })
    >>> value, details = divisiveness_simple(df, method="winrate")
    """
    data = data.copy()
    
    # Get the aggregation function
    if isinstance(method, str):
        if method not in AGGREGATION_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Available methods: "
                f"{list(AGGREGATION_METHODS.keys())}"
            )
        agg_func = AGGREGATION_METHODS[method]
    else:
        agg_func = method
    
    # Get all unique proposals
    all_proposals = sorted(
        set(data[proposal].unique()) | set(data[wins_over].unique())
    )
    N = len(all_proposals)
    
    if N < 2:
        return 0.0, pd.DataFrame({proposal: all_proposals, "divisiveness": [0.0] * N})
    
    # Pre-aggregate data
    pair_votes = data.groupby([proposal, wins_over])[count].sum().reset_index()
    
    # Compute global scores using the aggregation method
    global_scores = agg_func(
        pair_votes, 
        proposal=proposal, 
        wins_over=wins_over, 
        count=count
    )
    global_scores_dict = dict(zip(global_scores[proposal], global_scores["score"]))
    
    # Normalize global scores to [0, 1] range for comparison
    max_score = max(global_scores_dict.values()) if global_scores_dict else 1
    min_score = min(global_scores_dict.values()) if global_scores_dict else 0
    score_range = max_score - min_score if max_score != min_score else 1
    
    divisiveness_values = {}
    
    for prop_i in all_proposals:
        squared_diffs_sum = 0.0
        
        for prop_j in all_proposals:
            if prop_i == prop_j:
                continue
            
            # Votes for P_i > P_j (population that preferred i over j)
            votes_i_over_j = pair_votes[
                (pair_votes[proposal] == prop_i) & (pair_votes[wins_over] == prop_j)
            ][count].sum()
            
            # Votes for P_j > P_i (population that preferred j over i)
            votes_j_over_i = pair_votes[
                (pair_votes[proposal] == prop_j) & (pair_votes[wins_over] == prop_i)
            ][count].sum()
            
            total_votes = votes_i_over_j + votes_j_over_i
            
            if total_votes > 0:
                # Create weighted subsets to compute scores for each subpopulation
                # Subpopulation 1: Those who prefer i over j
                subset_i = pair_votes.copy()
                subset_i[count] = subset_i.apply(
                    lambda r: r[count] if (r[proposal] == prop_i and r[wins_over] == prop_j) or 
                              (r[proposal] != prop_j or r[wins_over] != prop_i) else 0, axis=1
                )
                subset_i = subset_i[subset_i[count] > 0]
                
                # Subpopulation 2: Those who prefer j over i
                subset_j = pair_votes.copy()
                subset_j[count] = subset_j.apply(
                    lambda r: r[count] if (r[proposal] == prop_j and r[wins_over] == prop_i) or 
                              (r[proposal] != prop_i or r[wins_over] != prop_j) else 0, axis=1
                )
                subset_j = subset_j[subset_j[count] > 0]
                
                # Compute score of prop_i in each subpopulation
                if len(subset_i) > 0:
                    scores_i = agg_func(subset_i, proposal=proposal, wins_over=wins_over, count=count)
                    score_i_in_subset_i = scores_i[scores_i[proposal] == prop_i]["score"].values
                    score_i_in_subset_i = (score_i_in_subset_i[0] - min_score) / score_range if len(score_i_in_subset_i) > 0 else 0
                else:
                    score_i_in_subset_i = 0
                
                if len(subset_j) > 0:
                    scores_j = agg_func(subset_j, proposal=proposal, wins_over=wins_over, count=count)
                    score_i_in_subset_j = scores_j[scores_j[proposal] == prop_i]["score"].values
                    score_i_in_subset_j = (score_i_in_subset_j[0] - min_score) / score_range if len(score_i_in_subset_j) > 0 else 0
                else:
                    score_i_in_subset_j = 0
                
                # The difference represents how differently the two subpopulations
                # view proposal i
                diff = score_i_in_subset_i - score_i_in_subset_j
                squared_diffs_sum += diff ** 2
        
        divisiveness_values[prop_i] = np.sqrt(squared_diffs_sum) / (N - 1)
    
    result_df = pd.DataFrame({
        proposal: list(divisiveness_values.keys()),
        "divisiveness": list(divisiveness_values.values())
    })
    result_df = result_df.sort_values("divisiveness", ascending=False).reset_index(drop=True)
    
    mean_divisiveness = result_df["divisiveness"].mean()
    
    return mean_divisiveness, result_df
