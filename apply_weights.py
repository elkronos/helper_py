import pandas as pd

def apply_weights(data, demographic_vars, demographic_weights, 
                  columns_to_weight=None, new_column_names=None, 
                  weight_suffix="_weight", value_suffix="_weighted", verbose=False):
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

    Returns
    -------
    pd.DataFrame
        Data frame with weighted columns for each demographic variable.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'x': [1, 2, 3, 4, 5],
    ...     'y': [6, 7, 8, 9, 10],
    ...     'gender': ["male", "female", "male", "male", "female"],
    ...     'race': ["white", "black", "white", "black", "white"]
    ... })
    >>> demographic_weights = {
    ...     'gender': {'male': 0.5, 'female': 1.5},
    ...     'race': {'white': 1.2, 'black': 0.8}
    ... }
    >>> result = apply_weights(data, ["gender", "race"], demographic_weights, verbose=True)
    >>> print(result)
    """
    _check_input(data, demographic_vars, demographic_weights, columns_to_weight, new_column_names)

    if columns_to_weight is None:
        columns_to_weight = data.select_dtypes(include=["number"]).columns.tolist()
    
    data = data.dropna().copy()
    weighted_data = data.copy()

    if new_column_names:
        if len(new_column_names) != len(columns_to_weight):
            raise ValueError("Length of new_column_names must equal length of columns_to_weight")
        weighted_column_names = new_column_names
    else:
        weighted_column_names = [f"{col}{value_suffix}" for col in columns_to_weight]
    
    for col, new_col_name in zip(columns_to_weight, weighted_column_names):
        total_weights = pd.Series([1] * len(data), index=data.index)
        for demographic_var in demographic_vars:
            if demographic_var in demographic_weights:
                level_weights = data[demographic_var].map(demographic_weights[demographic_var])
                if verbose:
                    weighted_data[f"{col}_{demographic_var}{weight_suffix}"] = level_weights
                total_weights *= level_weights
        weighted_data[new_col_name] = data[col] * total_weights
        
    return weighted_data


def _check_input(data, demographic_vars, demographic_weights, columns_to_weight, new_column_names):
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
    
    for var, weights in demographic_weights.items():
        if var in data.columns:
            unique_levels = pd.unique(data[var])
            if not all(level in weights for level in unique_levels):
                raise ValueError(f"All levels of {var} must have a corresponding weight")