import pandas as pd
import logging
import time
import functools
from typing import List, Optional, Callable, Union, Literal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def profile(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def validate_columns(df: pd.DataFrame, columns: Optional[Union[List[str], str]], param_name: str):
    if columns is None:
        return None
    columns = [columns] if isinstance(columns, str) else columns
    invalid_columns = [col for col in columns if col not in df.columns]
    if invalid_columns:
        raise ValueError(f"Invalid {param_name} column names: {invalid_columns}")
    return columns

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
    """
    logging.info("Starting deduplication process")
    columns_to_check = validate_columns(df, columns_to_check, "columns_to_check")
    group_by = validate_columns(df, group_by, "group_by")

    if action not in ['remove', 'flag']:
        raise ValueError("Invalid action. Choose 'remove' or 'flag'.")
    if tie_breaker not in ['keep_first', 'keep_last', 'random', 'custom']:
        raise ValueError("Invalid tie_breaker. Choose 'keep_first', 'keep_last', 'random', or 'custom'.")
    if tie_breaker == 'custom' and (custom_tie_breaker is None or not callable(custom_tie_breaker)):
        raise ValueError("custom_tie_breaker must be a callable function when tie_breaker is 'custom'.")

    if action == 'remove':
        return _remove_duplicates(df, columns_to_check, tie_breaker, group_by, custom_tie_breaker)
    return _flag_duplicates(df, columns_to_check, flag_column, group_by)

def _apply_tie_breaker(group: pd.DataFrame, tie_breaker: str, custom_tie_breaker: Optional[Callable]) -> pd.DataFrame:
    """Applies tie-breaker logic to resolve duplicates."""
    if tie_breaker == 'keep_first':
        return group.head(1)
    if tie_breaker == 'keep_last':
        return group.tail(1)
    if tie_breaker == 'random':
        return group.sample(n=1, random_state=42)
    return custom_tie_breaker(group)

def _remove_duplicates(df: pd.DataFrame, columns_to_check: Optional[List[str]], tie_breaker: str, 
                       group_by: Optional[List[str]], custom_tie_breaker: Optional[Callable]) -> pd.DataFrame:
    """Removes duplicate rows based on specified columns and tie-breaking criteria."""
    logging.info("Removing duplicates...")
    if columns_to_check is None:
        columns_to_check = df.columns.tolist()
    
    if group_by is None:
        return df.drop_duplicates(subset=columns_to_check, keep='first' if tie_breaker == 'keep_first' else 'last')
    
    return df.groupby(group_by, group_keys=False).apply(
        lambda x: _apply_tie_breaker(x.drop_duplicates(subset=columns_to_check), tie_breaker, custom_tie_breaker)
    ).reset_index(drop=True)

def _flag_duplicates(df: pd.DataFrame, columns_to_check: Optional[List[str]], flag_column: str, 
                     group_by: Optional[List[str]]) -> pd.DataFrame:
    """Flags duplicate rows instead of removing them."""
    df = df.copy()
    logging.info("Flagging duplicates...")
    if columns_to_check is None:
        columns_to_check = df.columns.tolist()
    
    if group_by is None:
        df[flag_column] = df.duplicated(subset=columns_to_check, keep=False)
    else:
        df[flag_column] = df.groupby(group_by, group_keys=False).apply(
            lambda x: x.duplicated(subset=columns_to_check, keep=False)
        ).reset_index(level=0, drop=True)
    return df

# Example usage
data = {
    'A': [1, 2, 2, 3, 4, 4, 1, 2, 2, 3, 4, 4],
    'B': ['a', 'b', 'b', 'c', 'd', 'd', 'a', 'b', 'b', 'c', 'd', 'd'],
    'C': [10, 20, 20, 30, 40, 40, 11, 21, 21, 31, 41, 41]
}
df = pd.DataFrame(data)

# Remove duplicates based on all columns
df_deduped = deduplicate_dataframe(df)
logging.info("Default deduplication:")
print(df_deduped)

# Flag duplicates instead of removing
df_flagged = deduplicate_dataframe(df, action='flag')
logging.info("Flagging duplicates:")
print(df_flagged)

# Custom tie-breaking criteria
def custom_criteria(group: pd.DataFrame) -> pd.DataFrame:
    return group.nlargest(1, 'C')

df_custom_tie = deduplicate_dataframe(df, columns_to_check=['A', 'B'], tie_breaker='custom', custom_tie_breaker=custom_criteria)
logging.info("Custom tie-breaking criteria:")
print(df_custom_tie)

# Group by 'A' and deduplicate within each group
df_grouped_deduped = deduplicate_dataframe(df, group_by='A', columns_to_check=['B'])
logging.info("Group by 'A' and deduplicate within each group:")
print(df_grouped_deduped)
