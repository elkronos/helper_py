import pandas as pd
import numpy as np

def apply_weights(data, demographic_vars, demographic_weights, 
                  columns_to_weight=None, new_column_names=None, 
                  weight_suffix="_weight", value_suffix="_weighted", 
                  verbose=False, handle_na='drop'):
    """
    Applies demographic weights to numeric columns in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The data frame to which weights are applied.
    demographic_vars : list[str]
        Names of demographic variables to apply weights to.
    demographic_weights : dict
        Dictionary containing demographic weights for each variable.
        E.g., {'gender': {'male': 0.5, 'female': 1.5}, 'race': {'white': 1.2, 'black': 0.8}}
    columns_to_weight : list[str], optional
        Names of numeric columns to apply weights to. 
        Applies to all numeric columns if None. Defaults to None.
    new_column_names : list[str], optional
        New column names for the weighted values. Must be of the same length
        as columns_to_weight if provided. Defaults to None.
    weight_suffix : str, optional
        Suffix for weighted column names. Defaults to "_weight".
    value_suffix : str, optional
        Suffix for total weighted column names. Defaults to "_weighted".
    verbose : bool, optional
        Include weights for each demographic variable in the output. Defaults to False.
    handle_na : str, optional
        How to handle NaN values. Options are 'drop', 'fill', or 'ignore'. Defaults to 'drop'.

    Returns
    -------
    pd.DataFrame
        Data frame with weighted columns for each demographic variable.
    """
    _check_input(data, demographic_vars, demographic_weights, columns_to_weight, new_column_names, handle_na)

    # Create a copy of the input data to avoid modifying the original
    data = data.copy()

    if columns_to_weight is None:
        columns_to_weight = data.select_dtypes(include=["number"]).columns.tolist()
    
    # Handle NaN values
    if handle_na == 'drop':
        data = data.dropna()
    elif handle_na == 'fill':
        # Fill numeric columns with mean, categorical columns with mode
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].mean())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
    # If 'ignore', we do nothing and keep NaN values

    weighted_data = data.copy()

    if new_column_names:
        if len(new_column_names) != len(columns_to_weight):
            raise ValueError("Length of new_column_names must equal length of columns_to_weight")
        weighted_column_names = new_column_names
    else:
        weighted_column_names = [f"{col}{value_suffix}" for col in columns_to_weight]
    
    for col, new_col_name in zip(columns_to_weight, weighted_column_names):
        total_weights = pd.Series(np.ones(len(data)), index=data.index)
        for demographic_var in demographic_vars:
            if demographic_var in demographic_weights:
                level_weights = data[demographic_var].map(lambda x: demographic_weights[demographic_var].get(x, 1.0))
                if verbose:
                    weighted_data[f"{col}_{demographic_var}{weight_suffix}"] = level_weights
                total_weights *= level_weights
        weighted_data[new_col_name] = data[col] * total_weights
        
    return weighted_data


def _check_input(data, demographic_vars, demographic_weights, columns_to_weight, new_column_names, handle_na):
    """
    Checks input types and values, raising a TypeError or ValueError if invalid.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame")
    if not isinstance(demographic_vars, list) or not all(isinstance(var, str) for var in demographic_vars):
        raise TypeError("demographic_vars must be a list of strings")
    if not isinstance(demographic_weights, dict) or not all(isinstance(weight, dict) for weight in demographic_weights.values()):
        raise TypeError("demographic_weights must be a dict of dict")
    if columns_to_weight is not None and (not isinstance(columns_to_weight, list) or not all(isinstance(col, str) for col in columns_to_weight)):
        raise TypeError("columns_to_weight must be None or a list of strings")
    if new_column_names is not None and (not isinstance(new_column_names, list) or not all(isinstance(col, str) for col in new_column_names)):
        raise TypeError("new_column_names must be None or a list of strings")
    if handle_na not in ['drop', 'fill', 'ignore']:
        raise ValueError("handle_na must be 'drop', 'fill', or 'ignore'")
    
    for var, weights in demographic_weights.items():
        if var in data.columns:
            unique_levels = pd.unique(data[var].dropna())
            if not all(level in weights for level in unique_levels):
                raise ValueError(f"All non-NaN levels of {var} must have a corresponding weight")
        if not all(isinstance(weight, (int, float)) for weight in weights.values()):
            raise TypeError(f"All weights for {var} must be numeric")

# Test the function with the same data as before

# Create a sample DataFrame with NaN values
data_with_nan = pd.DataFrame({
    'income': [50000, 60000, 75000, 80000, 90000, 100000],
    'age': [25, 30, 35, 40, 45, 50],
    'gender': ['male', 'female', 'male', 'female', 'male', 'female'],
    'education': ['high school', 'bachelor', np.nan, 'bachelor', 'phd', 'master']
})

# Set the first income value to NaN
data_with_nan.loc[0, 'income'] = np.nan

# Define demographic weights
demographic_weights = {
    'gender': {'male': 0.8, 'female': 1.2},
    'education': {'high school': 1.5, 'bachelor': 1.0, 'master': 0.8, 'phd': 0.7}
}

print("Data with NaN values:")
print(data_with_nan)

# Apply weights with different NaN handling options
result_drop = apply_weights(data_with_nan, ['gender', 'education'], demographic_weights, handle_na='drop', verbose=True)
result_fill = apply_weights(data_with_nan, ['gender', 'education'], demographic_weights, handle_na='fill', verbose=True)
result_ignore = apply_weights(data_with_nan, ['gender', 'education'], demographic_weights, handle_na='ignore', verbose=True)

print("\nWeighted Data (NaN dropped):")
print(result_drop)
print("\nWeighted Data (NaN filled):")
print(result_fill)
print("\nWeighted Data (NaN ignored):")
print(result_ignore)