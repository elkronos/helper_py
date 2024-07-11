import pandas as pd
import logging
import time
import functools
from typing import List, Optional, Callable, Union, Literal

logging.basicConfig(level=logging.INFO)

def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@profile
def deduplicate_dataframe(df: pd.DataFrame,
                          columns_to_check: Optional[Union[List[str], str]] = None,
                          action: Literal['remove', 'flag'] = 'remove',
                          tie_breaker: Literal['keep_first', 'keep_last', 'random', 'custom'] = 'keep_first',
                          flag_column: str = 'is_duplicate',
                          custom_tie_breaker: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                          group_by: Optional[Union[List[str], str]] = None) -> pd.DataFrame:
    """
    Deduplicate a DataFrame based on specified columns, group, and tie-breaking criteria.

    Parameters:
    - df (pd.DataFrame): The DataFrame to deduplicate.
    - columns_to_check (list, str or None): Columns to check for duplicates. Defaults to None, which means all columns.
    - action (str): Action to perform on duplicates. Options are 'remove' or 'flag'. Defaults to 'remove'.
    - tie_breaker (str): Criteria to break ties. Options are 'keep_first', 'keep_last', 'random', or 'custom'. Defaults to 'keep_first'.
    - flag_column (str): Name of the flag column to indicate duplicates when action is 'flag'. Defaults to 'is_duplicate'.
    - custom_tie_breaker (function or None): Custom function to resolve duplicates. Only used if tie_breaker is 'custom'.
    - group_by (list, str or None): Columns to group by before checking for duplicates. Defaults to None, meaning no grouping.

    Returns:
    - pd.DataFrame: Deduplicated DataFrame.

    Raises:
    - ValueError: If an invalid column name or action is provided.
    """
    print(f"Initial DataFrame:\n{df}")
    validate_params(df, columns_to_check, action, tie_breaker, custom_tie_breaker, group_by)

    if action == 'remove':
        result = remove_duplicates(df, columns_to_check, tie_breaker, group_by, custom_tie_breaker)
    elif action == 'flag':
        result = flag_duplicates(df, columns_to_check, flag_column, group_by)

    logging.info("Deduplication completed successfully.")
    print(f"Resultant DataFrame after deduplication:\n{result}")
    return result

def validate_params(df: pd.DataFrame,
                    columns_to_check: Optional[Union[List[str], str]],
                    action: str,
                    tie_breaker: str,
                    custom_tie_breaker: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
                    group_by: Optional[Union[List[str], str]]):
    print(f"Validating parameters...")
    columns_to_check_list = [columns_to_check] if isinstance(columns_to_check, str) else columns_to_check
    group_by_list = [group_by] if isinstance(group_by, str) else group_by

    if columns_to_check_list is not None:
        invalid_cols = [col for col in columns_to_check_list if col not in df.columns]
        if invalid_cols:
            raise ValueError(f"Invalid column names: {invalid_cols}")

    if group_by_list is not None:
        invalid_group_cols = [col for col in group_by_list if col not in df.columns]
        if invalid_group_cols:
            raise ValueError(f"Invalid group column names: {invalid_group_cols}")

    if action not in ['remove', 'flag']:
        raise ValueError("Invalid action. Options are 'remove' or 'flag'.")

    if tie_breaker not in ['keep_first', 'keep_last', 'random', 'custom']:
        raise ValueError("Invalid tie_breaker. Options are 'keep_first', 'keep_last', 'random', or 'custom'.")

    if tie_breaker == 'custom' and (custom_tie_breaker is None or not callable(custom_tie_breaker)):
        raise ValueError("custom_tie_breaker must be a callable function when tie_breaker is 'custom'.")

def _apply_tie_breaker(group: pd.DataFrame, tie_breaker: str, custom_tie_breaker: Optional[Callable[[pd.DataFrame], pd.DataFrame]]) -> pd.DataFrame:
    print(f"Applying tie breaker '{tie_breaker}' on group:\n{group}")
    if tie_breaker == 'keep_first':
        result = group.head(1)
    elif tie_breaker == 'keep_last':
        result = group.tail(1)
    elif tie_breaker == 'random':
        result = group.sample(n=1)
    elif tie_breaker == 'custom':
        result = custom_tie_breaker(group)
    print(f"Result after applying tie breaker:\n{result}")
    return result

def remove_duplicates(df: pd.DataFrame, 
                      columns_to_check: Optional[Union[List[str], str]], 
                      tie_breaker: str, 
                      group_by: Optional[Union[List[str], str]], 
                      custom_tie_breaker: Optional[Callable[[pd.DataFrame], pd.DataFrame]]) -> pd.DataFrame:
    if columns_to_check is None:
        columns_to_check = df.columns.tolist()
    
    print(f"Removing duplicates with columns_to_check={columns_to_check}, tie_breaker={tie_breaker}, group_by={group_by}")
    if group_by is None:
        if tie_breaker == 'custom':
            df = df.sort_values(by=columns_to_check)
            deduplicated_df = df.groupby(columns_to_check).apply(custom_tie_breaker).reset_index(drop=True)
        else:
            deduplicated_df = df.drop_duplicates(subset=columns_to_check, keep='first' if tie_breaker == 'keep_first' else 'last')
    else:
        grouped = df.groupby(group_by)
        deduplicated_df = grouped.apply(lambda x: _apply_tie_breaker(x.drop_duplicates(subset=columns_to_check), tie_breaker, custom_tie_breaker)).reset_index(drop=True)
    
    print(f"Deduplicated DataFrame:\n{deduplicated_df}")
    return deduplicated_df

def flag_duplicates(df: pd.DataFrame, 
                    columns_to_check: Optional[Union[List[str], str]], 
                    flag_column: str, 
                    group_by: Optional[Union[List[str], str]]) -> pd.DataFrame:
    df_copy = df.copy()
    if columns_to_check is None:
        columns_to_check = df.columns.tolist()
    
    print(f"Flagging duplicates with columns_to_check={columns_to_check}, group_by={group_by}")
    if group_by is None:
        df_copy[flag_column] = df_copy.duplicated(subset=columns_to_check, keep=False)
    else:
        df_copy[flag_column] = df_copy.groupby(group_by).apply(lambda x: x.duplicated(subset=columns_to_check, keep=False)).reset_index(level=0, drop=True)
    
    print(f"DataFrame after flagging duplicates:\n{df_copy}")
    return df_copy

# Example usage
if __name__ == "__main__":
    data = {
        'A': [1, 2, 2, 3, 4, 4, 1, 2, 2, 3, 4, 4],
        'B': ['a', 'b', 'b', 'c', 'd', 'd', 'a', 'b', 'b', 'c', 'd', 'd'],
        'C': [10, 20, 20, 30, 40, 40, 11, 21, 21, 31, 41, 41]
    }
    df = pd.DataFrame(data)

    # Remove duplicates based on all columns
    df_deduped = deduplicate_dataframe(df)
    print("Default deduplication:")
    print(df_deduped)

    # Remove duplicates based on specific columns
    df_deduped_cols = deduplicate_dataframe(df, columns_to_check=['A', 'B'])
    print("\nDeduplication based on specific columns:")
    print(df_deduped_cols)

    # Flag duplicates instead of removing
    df_flagged = deduplicate_dataframe(df, action='flag')
    print("\nFlagging duplicates:")
    print(df_flagged)

    # Custom tie-breaking criteria
    def custom_criteria(group: pd.DataFrame) -> pd.DataFrame:
        return group.nlargest(1, 'C')
    
    df_custom_tie = deduplicate_dataframe(df, columns_to_check=['A', 'B'], tie_breaker='custom', custom_tie_breaker=custom_criteria)
    print("\nCustom tie-breaking criteria:")
    print(df_custom_tie)
    
    # Group by 'A' and deduplicate within each group
    df_grouped_deduped = deduplicate_dataframe(df, group_by='A', columns_to_check=['B'])
    print("\nGroup by 'A' and deduplicate within each group:")
    print(df_grouped_deduped)
