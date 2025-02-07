import pandas as pd
import logging
import time
import functools
from typing import List, Optional, Callable, Union, Literal

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def profile(func: Callable) -> Callable:
    """Decorator to log the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"Function {func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper


@profile
def deduplicate_dataframe(
    df: pd.DataFrame,
    columns_to_check: Optional[Union[List[str], str]] = None,
    action: Literal['remove', 'flag'] = 'remove',
    tie_breaker: Literal['keep_first', 'keep_last', 'random', 'custom'] = 'keep_first',
    flag_column: str = 'is_duplicate',
    custom_tie_breaker: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    group_by: Optional[Union[List[str], str]] = None
) -> pd.DataFrame:
    """
    Deduplicate a DataFrame based on specified columns, optional grouping, and tie-breaking criteria.

    Args:
        df (pd.DataFrame): DataFrame to deduplicate.
        columns_to_check (List[str] or str, optional): Columns to check for duplicates.
            Defaults to None (i.e. all columns).
        action (str): 'remove' to drop duplicates, 'flag' to add a flag column.
        tie_breaker (str): Tie-breaking method: 'keep_first', 'keep_last', 'random', or 'custom'.
            Defaults to 'keep_first'.
        flag_column (str): Column name to flag duplicates if action is 'flag'.
        custom_tie_breaker (Callable, optional): Function to resolve duplicates when tie_breaker is 'custom'.
        group_by (List[str] or str, optional): Columns to group by before checking for duplicates.

    Returns:
        pd.DataFrame: The deduplicated DataFrame.

    Raises:
        ValueError: If any parameter is invalid.
    """
    logger.debug("Starting deduplication process.")
    validate_params(df, columns_to_check, action, tie_breaker, custom_tie_breaker, group_by)

    original_count = len(df)
    if action == 'remove':
        deduped_df = remove_duplicates(df, columns_to_check, tie_breaker, group_by, custom_tie_breaker)
        final_count = len(deduped_df)
        stats = {
            'original_count': original_count,
            'final_count': final_count,
            'duplicates_removed': original_count - final_count
        }
        logger.info(f"Deduplication stats: {stats}")
    elif action == 'flag':
        deduped_df = flag_duplicates(df, columns_to_check, flag_column, group_by)
        # Count flagged duplicates only if flag_column is boolean
        flagged_count = int(deduped_df[flag_column].sum()) if deduped_df[flag_column].dtype == 'bool' else None
        stats = {
            'original_count': original_count,
            'flagged_duplicates': flagged_count
        }
        logger.info(f"Flagging stats: {stats}")
    else:
        # Should not happen due to validation
        raise ValueError(f"Invalid action: {action}")

    logger.debug("Deduplication process completed.")
    return deduped_df


def validate_params(
    df: pd.DataFrame,
    columns_to_check: Optional[Union[List[str], str]],
    action: str,
    tie_breaker: str,
    custom_tie_breaker: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
    group_by: Optional[Union[List[str], str]]
) -> None:
    """
    Validate parameters for deduplication.

    Raises:
        ValueError: If any parameter is invalid.
    """
    logger.debug("Validating parameters.")

    if columns_to_check is not None:
        if isinstance(columns_to_check, str):
            columns_to_check_list = [columns_to_check]
        elif isinstance(columns_to_check, list):
            columns_to_check_list = columns_to_check
        else:
            raise ValueError("columns_to_check must be a string or a list of strings.")
        invalid_cols = [col for col in columns_to_check_list if col not in df.columns]
        if invalid_cols:
            raise ValueError(f"Invalid column names in columns_to_check: {invalid_cols}")

    if group_by is not None:
        if isinstance(group_by, str):
            group_by_list = [group_by]
        elif isinstance(group_by, list):
            group_by_list = group_by
        else:
            raise ValueError("group_by must be a string or a list of strings.")
        invalid_group_cols = [col for col in group_by_list if col not in df.columns]
        if invalid_group_cols:
            raise ValueError(f"Invalid column names in group_by: {invalid_group_cols}")

    if action not in ['remove', 'flag']:
        raise ValueError("action must be 'remove' or 'flag'.")

    if tie_breaker not in ['keep_first', 'keep_last', 'random', 'custom']:
        raise ValueError("tie_breaker must be 'keep_first', 'keep_last', 'random', or 'custom'.")

    if tie_breaker == 'custom' and (custom_tie_breaker is None or not callable(custom_tie_breaker)):
        raise ValueError("A callable custom_tie_breaker must be provided when tie_breaker is 'custom'.")


def _apply_tie_breaker(
    group: pd.DataFrame,
    tie_breaker: str,
    custom_tie_breaker: Optional[Callable[[pd.DataFrame], pd.DataFrame]]
) -> pd.DataFrame:
    """
    Apply the tie-breaking criteria to a group of duplicates.

    Args:
        group (pd.DataFrame): Group of duplicate rows.
        tie_breaker (str): Tie-breaking method.
        custom_tie_breaker (Callable, optional): Custom function if tie_breaker is 'custom'.

    Returns:
        pd.DataFrame: A DataFrame containing the selected row(s) from the group.
    """
    logger.debug(f"Applying tie breaker '{tie_breaker}' on group with shape {group.shape}.")
    if tie_breaker == 'keep_first':
        result = group.head(1)
    elif tie_breaker == 'keep_last':
        result = group.tail(1)
    elif tie_breaker == 'random':
        result = group.sample(n=1)
    elif tie_breaker == 'custom':
        result = custom_tie_breaker(group)
    else:
        raise ValueError(f"Unsupported tie_breaker: {tie_breaker}")
    logger.debug(f"Resulting group shape after tie breaker: {result.shape}.")
    return result


def remove_duplicates(
    df: pd.DataFrame,
    columns_to_check: Optional[Union[List[str], str]],
    tie_breaker: str,
    group_by: Optional[Union[List[str], str]],
    custom_tie_breaker: Optional[Callable[[pd.DataFrame], pd.DataFrame]]
) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame based on the specified criteria.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_check (List[str] or str, optional): Columns to check for duplicates.
            Defaults to all columns if None.
        tie_breaker (str): The tie-breaking method.
        group_by (List[str] or str, optional): Columns to group by before deduplication.
        custom_tie_breaker (Callable, optional): Custom function for tie breaking when required.

    Returns:
        pd.DataFrame: The deduplicated DataFrame.
    """
    logger.debug("Starting removal of duplicates.")
    if columns_to_check is None:
        columns_to_check = df.columns.tolist()
    elif isinstance(columns_to_check, str):
        columns_to_check = [columns_to_check]

    if group_by is None:
        if tie_breaker == 'custom':
            # When using a custom function, sort and then group to apply it.
            df_sorted = df.sort_values(by=columns_to_check)
            deduped_df = df_sorted.groupby(columns_to_check, as_index=False).apply(custom_tie_breaker)
            deduped_df.reset_index(drop=True, inplace=True)
        else:
            keep_option = 'first' if tie_breaker == 'keep_first' else 'last'
            deduped_df = df.drop_duplicates(subset=columns_to_check, keep=keep_option)
    else:
        if isinstance(group_by, str):
            group_by = [group_by]
        # Apply tie-breaker within each group defined by group_by.
        deduped_df = df.groupby(group_by, group_keys=False).apply(
            lambda group: _apply_tie_breaker(group, tie_breaker, custom_tie_breaker)
        ).reset_index(drop=True)

    logger.debug(f"Removed duplicates. New DataFrame shape: {deduped_df.shape}.")
    return deduped_df


def flag_duplicates(
    df: pd.DataFrame,
    columns_to_check: Optional[Union[List[str], str]],
    flag_column: str,
    group_by: Optional[Union[List[str], str]]
) -> pd.DataFrame:
    """
    Flag duplicate rows in the DataFrame by adding a boolean flag column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_check (List[str] or str, optional): Columns to check for duplicates.
            Defaults to all columns if None.
        flag_column (str): Name of the flag column.
        group_by (List[str] or str, optional): Columns to group by before flagging.

    Returns:
        pd.DataFrame: DataFrame with an additional column indicating duplicate rows.
    """
    logger.debug("Starting flagging of duplicates.")
    df_copy = df.copy()
    if columns_to_check is None:
        columns_to_check = df.columns.tolist()
    elif isinstance(columns_to_check, str):
        columns_to_check = [columns_to_check]

    if group_by is None:
        df_copy[flag_column] = df_copy.duplicated(subset=columns_to_check, keep=False)
    else:
        if isinstance(group_by, str):
            group_by = [group_by]

        def flag_group(group: pd.DataFrame) -> pd.Series:
            return group.duplicated(subset=columns_to_check, keep=False)

        # Apply flagging within each group.
        flagged = df_copy.groupby(group_by, group_keys=False).apply(lambda g: flag_group(g))
        # Ensure alignment with the original DataFrame index.
        df_copy[flag_column] = flagged.reset_index(drop=True)
    logger.debug("Flagging completed.")
    return df_copy


# Example usage
if __name__ == "__main__":
    # Set the logging level to DEBUG for detailed output during testing.
    logging.getLogger().setLevel(logging.DEBUG)

    data = {
        'A': [1, 2, 2, 3, 4, 4, 1, 2, 2, 3, 4, 4],
        'B': ['a', 'b', 'b', 'c', 'd', 'd', 'a', 'b', 'b', 'c', 'd', 'd'],
        'C': [10, 20, 20, 30, 40, 40, 11, 21, 21, 31, 41, 41]
    }
    df = pd.DataFrame(data)
    logger.info("Original DataFrame:")
    logger.info(df)

    # 1. Default deduplication based on all columns (remove duplicates)
    df_deduped = deduplicate_dataframe(df)
    logger.info("Deduplicated DataFrame (default):")
    logger.info(df_deduped)

    # 2. Deduplication based on specific columns ['A', 'B']
    df_deduped_cols = deduplicate_dataframe(df, columns_to_check=['A', 'B'])
    logger.info("Deduplicated DataFrame based on columns ['A', 'B']:")
    logger.info(df_deduped_cols)

    # 3. Flag duplicates instead of removing them
    df_flagged = deduplicate_dataframe(df, action='flag')
    logger.info("DataFrame with duplicates flagged:")
    logger.info(df_flagged)

    # 4. Custom tie-breaking: keep the row with the highest value in column 'C'
    def custom_criteria(group: pd.DataFrame) -> pd.DataFrame:
        if group.empty:
            return group
        # Use idxmax to select the row with the highest 'C'
        return group.loc[[group['C'].idxmax()]]

    df_custom_tie = deduplicate_dataframe(
        df,
        columns_to_check=['A', 'B'],
        tie_breaker='custom',
        custom_tie_breaker=custom_criteria
    )
    logger.info("DataFrame after applying custom tie-breaking:")
    logger.info(df_custom_tie)

    # 5. Group by column 'A' and deduplicate within each group based on column 'B'
    df_grouped_deduped = deduplicate_dataframe(df, group_by='A', columns_to_check=['B'])
    logger.info("DataFrame after grouping by 'A' and deduplication on 'B':")
    logger.info(df_grouped_deduped)
