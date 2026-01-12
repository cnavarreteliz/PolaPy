import pandas as pd
from typing import List, Union


def dhondt(
    input_df: pd.DataFrame,
    party: str = "party",
    votes: str = "votes",
    n_seats: Union[int, str] = 1,
    levels: Union[str, List[str]] = None,
    quota: bool = False
) -> pd.DataFrame:
    """D'Hondt (or Jefferson) method with multi-level support.

    Calculates the number of elected alternatives of each party using the 
    D'Hondt (or Jefferson) method. Supports hierarchical allocation across
    multiple levels (e.g., districts, regions, national).

    Parameters
    ----------
    input_df : pd.DataFrame
        Dataset with party votes. Must contain columns for party identifier,
        votes, and optionally level groupings.
    party : str, optional
        Column name for party/candidate identifiers, by default "party".
    votes : str, optional
        Column name for vote counts, by default "votes".
    n_seats : int or str, optional
        Number of seats to allocate. Can be:
        - An integer: same number of seats for all groups
        - A column name: use values from this column as seats per group
        By default 1.
    levels : str or List[str], optional
        Column name(s) defining the hierarchical levels for seat allocation.
        If None, allocation is done globally. If a list, seats are allocated
        within each unique combination of level values.
        Examples:
        - None: global allocation
        - "district": allocate seats within each district
        - ["region", "district"]: allocate seats within each region-district pair
        By default None.
    quota : bool, optional
        If True, returns the quota values along with allocations, by default False.

    Returns
    -------
    pd.DataFrame
        Summary with the seats of each party (and level groups if specified).
        If quota=True, returns a tuple (DataFrame with quotas, threshold quota value).

    References
    ----------
    D'Hondt, V. (1882). Système pratique et raisonné de représentation 
    proportionnelle. Muquardt.

    Examples
    --------
    >>> import pandas as pd
    >>> # Simple global allocation
    >>> df = pd.DataFrame({'party': ['A', 'B', 'C'], 'votes': [5000, 3000, 2000]})
    >>> dhondt(df, n_seats=5)
    
    >>> # Multi-district allocation
    >>> df = pd.DataFrame({
    ...     'district': ['D1', 'D1', 'D2', 'D2'],
    ...     'party': ['A', 'B', 'A', 'B'],
    ...     'votes': [1000, 800, 600, 900],
    ...     'seats': [2, 2]
    ... })
    >>> dhondt(df, levels='district', n_seats='seats')
    
    >>> # Multi-level hierarchical allocation
    >>> df = pd.DataFrame({
    ...     'region': ['R1', 'R1', 'R1', 'R1', 'R2', 'R2'],
    ...     'district': ['D1', 'D1', 'D2', 'D2', 'D3', 'D3'],
    ...     'party': ['A', 'B', 'A', 'B', 'A', 'B'],
    ...     'votes': [1000, 800, 600, 900, 700, 500],
    ...     'seats': [2, 2, 3]
    ... })
    >>> dhondt(df, levels=['region', 'district'], n_seats='seats')
    """
    input_df = input_df.copy()
    
    # Normalize levels to a list
    if levels is None:
        level_cols = []
    elif isinstance(levels, str):
        level_cols = [levels]
    else:
        level_cols = list(levels)
    
    # Determine grouping columns
    group_cols = level_cols + [party]
    
    def allocate_seats_in_group(group_df: pd.DataFrame, group_n_seats: int) -> pd.DataFrame:
        """Allocate seats within a single group using D'Hondt."""
        output = []
        for _, row in group_df.iterrows():
            __party__ = row[party]
            __votes__ = row[votes]
            
            # Include level values in output
            level_values = {col: row[col] for col in level_cols if col in row.index}
            
            for i in range(group_n_seats):
                entry = {
                    party: __party__,
                    "quota": __votes__ / (i + 1),
                    **level_values
                }
                output.append(entry)
        
        if not output:
            return pd.DataFrame()
        
        tmp = pd.DataFrame(output).sort_values("quota", ascending=False).reset_index(drop=True)
        return tmp, tmp.loc[group_n_seats - 1, "quota"] if group_n_seats > 0 and len(tmp) >= group_n_seats else (tmp, 0)
    
    # If no levels, do global allocation
    if not level_cols:
        # Get number of seats
        if isinstance(n_seats, str):
            group_n_seats = int(input_df[n_seats].iloc[0])
        else:
            group_n_seats = n_seats
        
        output = []
        for __party__, df_tmp in input_df.groupby(party):
            __votes__ = df_tmp[votes].sum()
            for i in range(group_n_seats):
                output.append({
                    party: __party__,
                    "quota": __votes__ / (i + 1)
                })
        
        tmp = pd.DataFrame(output).sort_values("quota", ascending=False).reset_index(drop=True)
        
        if quota:
            return tmp, tmp.loc[group_n_seats - 1, "quota"] if group_n_seats > 0 and len(tmp) >= group_n_seats else 0
        
        return tmp.head(group_n_seats).groupby(party).count().reset_index().rename(
            columns={"quota": "seats"}
        ).sort_values("seats", ascending=False)
    
    # Multi-level allocation
    all_quotas = []
    all_results = []
    quota_thresholds = []
    # Use string for single level groupby to avoid tuples
    groupby_key = level_cols[0] if len(level_cols) == 1 else level_cols
    
    for level_values, group_df in input_df.groupby(groupby_key):
        # Normalize level_values to list for consistent indexing
        if len(level_cols) == 1:
            level_values_list = [level_values]
        else:
            level_values_list = list(level_values)
        
        # Get number of seats for this group
        if isinstance(n_seats, str):
            group_n_seats = int(group_df[n_seats].iloc[0])
        else:
            group_n_seats = n_seats
        
        if group_n_seats <= 0:
            continue
        
        # Aggregate votes by party within this group
        group_agg = group_df.groupby(party, as_index=False)[votes].sum()
        
        # Generate quotas for each party
        output = []
        for _, row in group_agg.iterrows():
            __party__ = row[party]
            __votes__ = row[votes]
            
            for i in range(group_n_seats):
                entry = {party: __party__, "quota": __votes__ / (i + 1)}
                for j, col in enumerate(level_cols):
                    entry[col] = level_values_list[j]
                output.append(entry)
        
        if not output:
            continue
        
        tmp = pd.DataFrame(output).sort_values("quota", ascending=False).reset_index(drop=True)
        
        # Store quota threshold
        if len(tmp) >= group_n_seats:
            quota_thresholds.append({
                **{col: level_values_list[j] for j, col in enumerate(level_cols)},
                "quota_threshold": tmp.loc[group_n_seats - 1, "quota"]
            })
        
        if quota:
            all_quotas.append(tmp)
        
        # Get seat allocation for this group
        winners = tmp.head(group_n_seats)
        
        # Count seats per party in this group
        group_result = winners.groupby([party] + level_cols).size().reset_index(name="seats")
        all_results.append(group_result)
    
    if not all_results:
        if quota:
            return pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame()
    
    # Combine results from all groups
    final_result = pd.concat(all_results, ignore_index=True)
    final_result = final_result.sort_values(
        level_cols + ["seats"], 
        ascending=[True] * len(level_cols) + [False]
    ).reset_index(drop=True)
    
    if quota:
        all_quotas_df = pd.concat(all_quotas, ignore_index=True) if all_quotas else pd.DataFrame()
        quota_thresholds_df = pd.DataFrame(quota_thresholds) if quota_thresholds else pd.DataFrame()
        return all_quotas_df, quota_thresholds_df
    
    return final_result
