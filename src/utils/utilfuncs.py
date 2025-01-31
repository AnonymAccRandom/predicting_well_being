import os
import shutil
from collections import defaultdict
from typing import Union, Any

import pandas as pd

# Define nested typing aliases for nested dictionaries
NestedDict = dict[Any, dict]


def apply_name_mapping(
    features: list[str], name_mapping: NestedDict, prefix: bool
) -> list[str]:
    """
    Maps a list of features to their descriptive names based on the second-level name
    in the provided mapping dictionary.

    Args:
        features: A list of feature strings in the format "first_level_second_level".
        name_mapping: A dictionary containing the mappings for descriptive names.
                      The keys are first-level feature names, and the values are dictionaries
                      mapping second-level feature names to their descriptive names.
        prefix: If True, the feature strings include a prefix that should be used for mapping.

    Returns:
        list: A list of mapped second-level feature names or the original features if no mapping is found.
    """
    mapped_features = []
    for feature in features:
        try:
            if prefix:
                first_level, _, second_level = feature.partition("_")
                second_level_mapped = name_mapping[first_level][second_level]
                mapped_features.append(second_level_mapped)

            else:
                second_level_mapped = name_mapping["crit"][feature]
                mapped_features.append(second_level_mapped)

        except KeyError:  # if meta-cols are passed
            print(f"No mapping found for {feature} ")
            mapped_features.append(feature)

    return mapped_features


def format_df(
    df: pd.DataFrame,
    capitalize: bool = False,
    columns: list[str] = None,
    decimals: int = 2,
) -> pd.DataFrame:
    """
    Formats specified numerical columns of a DataFrame to the given number of decimal places.
    If no columns are specified, all numerical columns are formatted. Optionally, capitalizes
    column names and string values.

    Args:
        df: The DataFrame containing the data.
        capitalize: If True, capitalize the first word of each column name and string values.
        columns: List of column names to format. If None, all numerical columns are formatted.
        decimals: Number of decimal places to round to.

    Returns:
        pd.DataFrame: A DataFrame with the specified (or all numerical) columns rounded to the given number of decimal places.
    """
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    for column in columns:
        if column in df:
            df[column] = df[column].apply(
                lambda x: custom_round(x, decimals)
            )  # TODO: use normal round?

    if capitalize:
        df.columns = df.columns.str.capitalize()
        for col in df.select_dtypes(include=["object", "string"]):
            df[col] = df[col].str.capitalize()

    return df


def custom_round(value: float, decimals: int, max_decimals: int = 10) -> float:
    """
    Custom rounding method to round to the specified number of decimals.
    If the rounded result is zero, recursively increases the precision to avoid losing small values.

    Args:
        value: The value to round.
        decimals: Number of decimal places to round to.
        max_decimals: Maximum precision to prevent infinite recursion.

    Returns:
        float: Rounded value with adjusted precision for small numbers.
    """
    if pd.isna(value) or value == 0:
        return value

    rounded = round(value, decimals)

    if decimals == max_decimals:
        print(f"Max decimals reached for {value}")
        return 0

    if (
        rounded == 0 and decimals < max_decimals
    ):  # If rounded result is zero, increase precision
        return custom_round(value, decimals + 1, max_decimals)

    return rounded


def create_defaultdict(n_nesting: int, default_factory: Any = int) -> defaultdict:
    """
    Recursively creates a nested defaultdict with a specified level of nesting.

    Args:
        n_nesting (int): The number of nested levels for the defaultdict.
        default_factory (Any): The default value for the innermost level.
                               Defaults to `int`.

    Returns:
        defaultdict: A nested defaultdict with `n_nesting` levels.
    """
    if n_nesting < 1:
        raise ValueError("n_nesting must be at least 1")

    def nested_factory():
        return (
            default_factory
            if n_nesting == 1
            else create_defaultdict(n_nesting - 1, default_factory)
        )

    return defaultdict(nested_factory)


def defaultdict_to_dict(
    dct: Union[defaultdict, dict, NestedDict]
) -> Union[dict, NestedDict]:
    """
    Recursively converts a defaultdict into a standard Python dictionary.

    Args:
        dct: The input dictionary, which may be a defaultdict or a nested dictionary containing defaultdict objects.

    Returns:
        dict: A standard Python dictionary with no defaultdict objects.
    """
    if isinstance(dct, defaultdict):
        dct = {k: defaultdict_to_dict(v) for k, v in dct.items()}
    return dct


def separate_binary_continuous_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Separates the columns of a DataFrame into binary and continuous categories.

    Binary columns are defined as those containing only the values 0, 1, or NaN across all rows.
    Continuous columns include all other columns that do not meet the binary criteria.

    Args:
        df: A pandas DataFrame whose columns are to be categorized.

    Returns:
        tuple:
            - A list of binary column names.
            - A list of continuous column names, preserving their order in the DataFrame.
    """
    # Identify binary columns
    binary_cols = list(df.columns[(df.isin([0, 1]) | df.isna()).all(axis=0)])

    # Move columns with specific suffixes to continuous
    suffixes = ("_mean", "_sd", "_min", "_max")
    reclassified_cols = [col for col in binary_cols if col.endswith(suffixes)]
    binary_cols = [col for col in binary_cols if col not in reclassified_cols]

    # Define continuous columns
    continuous_cols = [col for col in df.columns if col not in binary_cols]

    return binary_cols, continuous_cols


def remove_leading_zero(value: Union[str, int, float]) -> str:
    """
    Removes the leading zero from a numeric value formatted as a string.

    Args:
        value: The input value, which can be a numeric or string type, or a pandas cell value.

    Returns:
        str: The input value as a string without a leading zero. If the input is not numeric, it is returned as-is.
    """
    try:
        # Convert value to string, remove leading zero if present
        value_str = str(value)

        if value_str.startswith("0") and len(value_str) > 1:
            return value_str.lstrip("0")

        if value_str.startswith("-0") and len(value_str) > 1:
            return value_str.lstrip("-0")

        return value_str

    except Exception as e:
        raise ValueError(f"Error processing value: {value}") from e


def merge_M_SD_into_cell(mean_series: pd.Series, sd_series: pd.Series) -> pd.Series:
    """
    Merges two pandas Series (Mean and Standard Deviation) into a single Series
    formatted as "M (SD)".

    Args:
        mean_series: A pandas Series containing the mean (M) values.
        sd_series: A pandas Series containing the standard deviation (SD) values.

    Returns:
        pd.Series: A Series where each element is formatted as "M (SD)".
    """
    return mean_series.astype(str) + " (" + sd_series.astype(str) + ")"


def merge_M_SD_in_dct(dct: Union[dict, NestedDict]) -> Union[dict, NestedDict]:
    """
    Recursively traverses a dictionary and merges "M" and "SD" keys at any level
    into a single key "M (SD)" with the format "xxx (yyy)".

    Args:
        dct: The dictionary to process.

    Returns:
        dict: The updated dictionary with "M" and "SD" merged into "M (SD)".
    """
    updated_dct = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            updated_dct[key] = merge_M_SD_in_dct(value)
        else:
            updated_dct[key] = value

    # Check if both "M" and "SD" keys exist in the current dictionary level
    if "M" in updated_dct and "SD" in updated_dct:
        mean = str(updated_dct.pop("M"))  # Get and remove "M"
        sd = str(updated_dct.pop("SD"))  # Get and remove "SD"
        updated_dct["M (SD)"] = f"{mean} ({sd})"

    return updated_dct


def format_p_values(lst_of_p_vals: list[float]) -> list[str]:
    """
    This function formats the p_values according to APA standards (3 decimals, <.001 otherwise)

    Args:
        lst_of_p_vals: list, containing the p_values for a given analysis setting.

    Returns:
        formatted_p_vals: list, contains p_values formatted according to APA style.
    """
    formatted_p_vals = []

    for p_val in lst_of_p_vals:
        if p_val < 0.001:
            formatted_p_vals.append("<.001")
        else:
            formatted = "{:.3f}".format(p_val).lstrip("0")
            formatted_p_vals.append(formatted)

    return formatted_p_vals


def rearrange_path_parts(
    root: str,
    base_dir: str,
    min_depth: int = 4,
    order_mapping: dict[str, int] = None,
    cat_values: dict[str, list[str]] = None,
) -> tuple[str, str, str, str, str] | None:
    """
    Rearranges parts of a relative path based on a mapping and possible values for categories.

    Args:
        root (str): The full path to process.
        base_dir (str): The base directory to calculate the relative path from.
        min_depth (int): Minimum depth of the path to proceed.
        order_mapping (dict[str, int]): Mapping of category names (e.g., 'crit') to their order in the rearranged key.
        cat_values (dict[str, list[str]]): Dictionary where keys are category names (e.g., 'crit') and values
                                           are lists of possible values for each category.

    Returns:
        tuple:
            - Rearranged path key as a string based on the mapping.
            - Values for crit, samples_to_include, feature_combination, and model as separate elements.
        None: If the depth requirement is not met.
    """
    # Normalize path and split into parts
    relative_path = os.path.relpath(root, base_dir)
    path_parts = relative_path.strip(os.sep).split(os.sep)

    if len(path_parts) < min_depth:
        print(f"Skipping directory {root} due to insufficient path depth.")
        return None

    # Determine the categories for each path part
    categorized_parts = {}
    for idx, part in enumerate(path_parts):
        for category, values in cat_values.items():
            if part in values:
                categorized_parts[category] = part

    # Rearrange based on the mapping
    try:
        rearranged_key = "_".join(categorized_parts[cat] for cat in order_mapping)
    except KeyError:
        print(
            "Key Error indicating that we try to process analyses we do not include in the tables"
        )
        return None

    return (
        rearranged_key,
        categorized_parts["crit"],
        categorized_parts["samples_to_include"],
        categorized_parts["feature_combination"],
        categorized_parts["model"],
    )


def inverse_code(df: pd.DataFrame, min_scale: int, max_scale: int) -> pd.DataFrame:
    """
    Performs inverse coding for items in a DataFrame.

    This method recodes items by subtracting each value from the sum of the scale's minimum
    and maximum values (`max_scale + min_scale`). It is typically used to reverse-code
    negative affect items.

    Args:
        df: A pandas DataFrame containing the items to be inverse-coded.
        min_scale: The minimum value of the scale.
        max_scale: The maximum value of the scale.

    Returns:
        pd.DataFrame: A DataFrame with inverse-coded values.
    """
    return max_scale + min_scale - df
