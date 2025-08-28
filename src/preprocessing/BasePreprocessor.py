import os
import pickle
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from typing import Union, Optional, Callable, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.utils.ConfigParser import ConfigParser
from src.utils.DataLoader import DataLoader
from src.utils.Logger import Logger
from src.utils.SanityChecker import SanityChecker
from src.utils.Timer import Timer
from src.utils.utilfuncs import NestedDict, inverse_code


class BasePreprocessor(ABC):
    """
    Abstract base class for preprocessing the different datasets.

    This class provides the foundation for preprocessing pipelines applied to different datasets.
    It includes methods for handling trait-level, ESM, sensing, and country-level data, as well as
    dataset-specific transformations implemented in subclasses. The preprocessing is modular and
    dynamically executed, with intermediate results logged and sanity-checked.
    It must be subclassed for implementation.

    **Basic Logic** (see also 'apply_preprocessing_methods'):
    1. Load raw data files for the dataset.
    2. Process and transform trait, ESM, sensing, and country-level data
    4. Merge data and execute final preprocessing steps on the combined data.

    Attributes:
        cfg_preprocessing (NestedDict): Yaml config specifying details on preprocessing (e.g., scales, items).
        dataset (str | None): The name of the dataset being processed. This is set in subclasses.
        logger (Logger): A logger for logging preprocessing steps and sanity checks
        timer (Timer): A timer to measure and log execution times for preprocessing methods.
        config_parser (Callable): Function to parse and retrieve configuration data from cfg_preprocessing.
        config_key_finder (Callable): Function to search for specific keys in the configuration.
        apply_preprocessing_methods (Callable): Preprocessing pipeline wrapped with a timer decorator.
        sanity_checker (SanityChecker): Validates the consistency and correctness of the processed data.
        data_loader (DataLoader): Utility to load raw data, optionally with a row limit for large files.
        df_before_final_selection (Optional[pd.DataFrame]): DataFrame containing data before final column selection.
    """

    def __init__(self, cfg_preprocessing: NestedDict) -> None:
        """
        Initializes the BasePreprocessor with a configuration file.

        Args:
            cfg_preprocessing: yaml config
        """
        self.cfg_preprocessing = cfg_preprocessing
        self.dataset = None  # assigned in the subclasses
        self.df_before_final_selection = None  # assigned during preprocessing

        self.data_loader = DataLoader(nrows=self.cfg_preprocessing["general"]["nrows"])
        self.logger = Logger(
            log_dir=self.cfg_preprocessing["general"]["log_dir"],
            log_file=self.cfg_preprocessing["general"]["log_name"],
        )
        self.timer = Timer(logger=self.logger)

        self.config_parser = ConfigParser().cfg_parser
        self.config_key_finder = ConfigParser().find_key_in_config

        self.apply_preprocessing_methods = self.timer._decorator(
            func=self.apply_preprocessing_methods
        )

        self.sanity_checker = SanityChecker(
            logger=self.logger,
            cfg_preprocessing=self.cfg_preprocessing,
            config_parser_class=ConfigParser(),
            apply_to_full_df=False,
        )

    @property
    def path_to_raw_data(self) -> str:
        """
        Constructs the full path to the folder containing the raw files for the specified dataset.

        Returns:
            str: The full path to the raw data folder for the dataset.
        """
        return os.path.join(
            self.cfg_preprocessing["general"]["path_to_raw_data"], self.dataset
        )

    @property
    def path_to_country_level_data(self) -> str:
        """
        Constructs the full path to the folder containing the raw country-level variable files.

        Returns:
            str: The full path to the folder containing the country-level variable files.
        """
        return os.path.join(
            self.cfg_preprocessing["general"]["path_to_raw_data"], "country_level_vars"
        )

    @property
    def path_to_sensing_data(self) -> Optional[str]:
        """
        Constructs the full path to the folder containing the sensing data,
        if the dataset is supported (e.g., "cocoms" or "zpid").

        Returns:
            Optional[str]: The full path to the sensing data folder if the dataset is supported;
            otherwise, None.
        """
        if self.dataset in ["cocoms", "zpid"]:
            return os.path.join(
                self.cfg_preprocessing["general"]["path_to_raw_data"],
                self.dataset,
                "sensing_vars",
            )
        else:
            return None

    @property
    def raw_trait_id_col(self) -> str:
        """
        Retrieves the dataset-specific Trait ID column. This must be accessed after the initialization
        of subclass-specific configurations.

        Returns:
            str: The Trait ID column name specific to the dataset.
        """
        return self.cfg_preprocessing["person_level"]["other_trait_columns"][0][
            "item_names"
        ][self.dataset]

    @property
    def raw_esm_id_col(self) -> str:
        """
        Retrieves the dataset-specific ESM ID column. This must be accessed after the initialization
        of subclass-specific configurations.

        Returns:
            str: The ESM ID column name specific to the dataset.
        """
        return self.cfg_preprocessing["esm_based"]["other_esm_columns"][0][
            "item_names"
        ][self.dataset]

    @property
    def raw_sensing_id_col(self) -> Optional[str]:
        """
        Retrieves the dataset-specific Sensing ID column. This must be accessed after the initialization
        of subclass-specific configurations. Only applicable for supported datasets (e.g., "cocoms" or "zpid").

        Returns:
            Optional[str]: The Sensing ID column name specific to the dataset if the dataset is supported;
            otherwise, None.
        """
        if self.dataset in ["cocoms", "zpid"]:
            return self.cfg_preprocessing["sensing_based"]["phone"][0]["item_names"][
                self.dataset
            ]
        else:
            return None

    @property
    def esm_timestamp(self) -> str:
        """
        Retrieves the dataset-specific ESM timestamp column. This must be accessed after the
        initialization of subclass-specific configurations.

        Returns:
            str: The ESM timestamp column name specific to the dataset.
        """
        return self.cfg_preprocessing["esm_based"]["other_esm_columns"][1][
            "item_names"
        ][self.dataset]

    def apply_preprocessing_methods(self) -> pd.DataFrame:
        """
        Executes the preprocessing pipeline for the current dataset, applying various preprocessing
        steps to trait-level, ESM, sensing, country-level data as well as the merged data for one dataset.
        Dataset-specific steps are defined in the subclasses.

        The pipeline involves:
        1. Loading raw data files for the dataset.
        2. Processing and transforming trait data (e.g., cleaning, aligning scales, creating binary variables).
        3. Processing and transforming ESM data (e.g., merging, creating pl-variables, collapsing data).
        4. Optionally processing sensing data for specific datasets (i.e., "cocoms" and "zpid").
        5. Merging country-level data into the dataset.
        6. Applying final preprocessing steps to the combined dataset, including dataset-specific post-processing,
           setting IDs, creating scales, and performing sanity checks.

        The preprocessing steps are modular and executed dynamically, with intermediate results logged and stored.

        Returns:
            pd.DataFrame: The fully preprocessed dataframe for the given dataset.
        """
        self.logger.log(f"--------------------------------------------------------")
        self.logger.log(f".")
        self.logger.log(f"Starting preprocessing pipeline for >>>{self.dataset}<<<")
        print(f"Starting preprocessing pipeline for >>>{self.dataset}<<<")

        # Step 1: Load data
        df_dct = self.data_loader.read_csv(path_to_dataset=self.path_to_raw_data)
        df_traits = None
        df_states = None

        # Step 2: Process trait data
        self.logger.log(f".")
        self.logger.log(f"  Preprocess trait-survey-based data")

        preprocess_steps_traits = [
            (self.merge_traits, {"df_dct": df_dct}),
            (self.clean_trait_col_duplicates, {"df_traits": None}),
            (self.exclude_flagged_rows, {"df_traits": None}),
            (self.adjust_education_level, {"df_traits": None}),
            (self.dataset_specific_trait_processing, {"df_traits": None}),
            (self.select_columns, {"df": None, "df_type": "person_level"}),
            (
                self.align_scales,
                {
                    "df": None,
                    "df_type": "person_level",
                    "cat_list": ["personality", "sociodemographics", "criterion"],
                },
            ),
            (self.check_scale_endpoints, {"df": None, "df_type": "person_level"}),
            (self.create_binary_vars_from_categoricals, {"df_traits": None}),
        ]

        for method, kwargs in preprocess_steps_traits:
            kwargs = {k: v if v is not None else df_traits for k, v in kwargs.items()}
            df_traits = self._log_and_execute(method, **kwargs)

        # Step 3: Process and transform esm data and merge on trait data
        self.logger.log(f".")
        self.logger.log(f"  Preprocess esm-based data")

        preprocess_steps_esm = [
            (self.merge_states, {"df_dct": df_dct}),
            (self.dataset_specific_state_processing, {"df_states": None}),
            (self.select_columns, {"df": None, "df_type": "esm_based"}),
            (
                self.align_scales,
                {
                    "df": None,
                    "df_type": "esm_based",
                    "cat_list": ["self_reported_micro_context", "criterion"],
                },
            ),
            (self.check_scale_endpoints, {"df": None, "df_type": "esm_based"}),
            (self.filter_min_num_esm_measures, {"df_states": None}),
            (self.store_wb_items, {"df_states": None}),
            (self.create_person_level_vars_from_esm, {"df_states": None}),
            (self.collapse_df, {"df": None, "df_type": "esm_based"}),
        ]

        for method, kwargs in preprocess_steps_esm:
            kwargs = {k: v if v is not None else df_states for k, v in kwargs.items()}
            df_states = self._log_and_execute(method, **kwargs)
        df_joint = self._log_and_execute(
            self.merge_dfs_on_id, **{"df_states": df_states, "df_traits": df_traits}
        )

        # Step 4 (optional): Specific sensed data processing
        if self.dataset in ["cocoms", "zpid"]:
            self.logger.log(f".")
            self.logger.log(f"  Preprocess sensed data")
            df_dct = self.data_loader.read_r(path_to_dataset=self.path_to_sensing_data)
            df_joint = self.process_and_merge_sensing_data(
                sensing_dct=df_dct, df=df_joint
            )

        # Step 5: Specific country data processing
        self.logger.log(f".")
        self.logger.log(f"  Preprocess country-level data")
        df_dct = self.data_loader.read_csv(
            path_to_dataset=self.path_to_country_level_data
        )
        df_joint = self.merge_country_data(country_var_dct=df_dct, df=df_joint)

        # Step 6: Merge data and joint processing
        self.logger.log(f".")
        self.logger.log(f"  Preprocess joint data")

        preprocess_steps_joint = [
            (self.dataset_specific_post_processing, {"df": None}),
            (self.set_id_as_index, {"df": None}),
            (self.inverse_coding, {"df": None}),
            (self.store_raw_trait_df, {"df": None}),
            (self.create_scale_means, {"df": None}),
            (self.store_trait_wb_items, {"df": None}),
            (self.create_criteria, {"df": None}),
            (self.set_full_col_df_as_attr, {"df": None}),
            (self.fill_unique_id_col, {"df": None}),
            (self.select_final_columns, {"df": None}),
            (self.sanity_checking, {"df": None}),
        ]

        for method, kwargs in preprocess_steps_joint:
            kwargs = {k: v if v is not None else df_joint for k, v in kwargs.items()}
            df_joint = self._log_and_execute(method, **kwargs)

        self.logger.log(".")
        self.logger.log(f"Finished preprocessing pipeline for >>>{self.dataset}<<<")
        self.logger.log(".")
        self.logger.log(f"--------------------------------------------------------")
        print(f"Finished preprocessing pipeline for >>>{self.dataset}<<<")

        return df_joint

    def _log_and_execute(
        self, method: Callable[..., Any], *args: Any, indent: int = 4, **kwargs: Any
    ) -> Any:
        """
        Logs and executes a specified method with optional positional and keyword arguments.

        Args:
            method: The method to be executed.
            indent: The number of spaces to prefix the log message with (default is 4).
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            Any: The result of the executed method (this is normally a pd.DataFrame in this class).
        """
        indent_spaces = " " * indent
        log_message = f"{indent_spaces}Executing {method.__name__}"

        self.logger.log(log_message)
        print(log_message)

        return method(*args, **kwargs)

    @abstractmethod
    def merge_traits(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges the raw data files containing trait data for the specified dataset.

        This method combines multiple dataframes found in `self.path_to_raw_data` into a single dataframe
        containing all trait variables for `self.dataset`. It must be implemented in the subclasses.

        Args:
            df_dct: A dictionary where the keys are filenames and the values
            are pandas DataFrames corresponding to the loaded data files.

        Returns:
            pd.DataFrame: A single DataFrame containing all merged trait data for the dataset.
        """
        pass

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        This method creates variable names corresponding to cfg_fix (i.e., it removes the timepoint suffixes
        from the column names and deletes the duplicate columns [e.g., _t1, _t2).

        Subclass implementations are optional, as not all datasets include variables with suffix.

        Args:
            df_traits: A pandas DataFrame containing the dataset.

        Returns:
            pd.DataFrame: A DataFrame with clean columns names
        """
        return df_traits

    def exclude_flagged_rows(self, df_traits) -> pd.DataFrame:
        """
        Excludes flagged rows from the trait DataFrame based on careless response indicators.

        This method removes rows in the trait data where individuals are flagged as having
        responded carelessly. The flags are defined in the configuration file and mapped to
        corresponding dataset-specific column names and categories.

        Args:
            df_traits: A pandas DataFrame containing the trait data.

        Returns:
            pd.DataFrame: A DataFrame with flagged rows excluded.
        """
        flags = self.config_parser(
            self.cfg_preprocessing["person_level"]["other_trait_columns"],
            "binary",
            "trait_flags",
        )[0]

        if self.dataset in flags["item_names"]:
            flag_col = flags["item_names"][self.dataset]
            df_traits[flag_col] = (
                df_traits[flag_col]
                .map(flags["category_mappings"][self.dataset])
                .fillna(1)
            )
            self.logger.log(
                f"      Persons in trait_df before excluding flagged samples: {df_traits[self.raw_trait_id_col].nunique()}"
            )

            df_traits = df_traits[df_traits[flag_col] == 1]
            self.logger.log(
                f"      Persons in trait_df after excluding flagged samples: {df_traits[self.raw_trait_id_col].nunique()}"
            )

        return df_traits

    def select_columns(
        self, df: pd.DataFrame, df_type: str = "person_level"
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only columns relevant to the specified dataset.

        This method identifies columns to retain based on the configuration in `self.cfg_preprocessing`,
        which specifies relevant column names for the given dataset.

        Args:
            df: A pandas DataFrame containing the dataset.
            df_type: The type of data to process. Can be "person_level", "esm_based", or "sensing_based".

        Returns:
            pd.DataFrame: A DataFrame filtered to include only the relevant columns.
        """
        if df_type in ["person_level", "esm_based"]:
            cols_to_be_selected = ["country"]
        else:
            cols_to_be_selected = []

        for cat, cat_entries in self.cfg_preprocessing[df_type].items():
            for entry in cat_entries:
                if "item_names" in entry:
                    cols_to_be_selected.extend(
                        self.extract_columns(entry["item_names"])
                    )

        df_col_filtered = df[cols_to_be_selected]
        df_col_filtered = df_col_filtered.loc[
            :, ~df_col_filtered.columns.duplicated()
        ].copy()

        return df_col_filtered

    def sort_dfs(self, df: pd.DataFrame, df_type: str = "person_level") -> pd.DataFrame:
        """
        Sorts the DataFrame based on the specified data type.

        - For "person_level", sorts the DataFrame by the raw trait ID column.
        - For "esm_based", sorts the DataFrame by the raw ESM ID column and timestamp.

        Args:
            df: A pandas DataFrame to be sorted.
            df_type: The type of data to sort. Can be "person_level" or "esm_based".

        Returns:
            pd.DataFrame: The sorted DataFrame.

        Raises:
            ValueError: If `df_type` is not "person_level" or "esm_based".
        """
        if df_type == "person_level":
            return df.sort_values(by=[self.raw_trait_id_col])

        elif df_type == "esm_based":
            return df.sort_values(by=[self.raw_esm_id_col, self.esm_timestamp])

        else:
            raise ValueError(
                f"Wrong df_type {df_type}, needs to be 'person_level' or 'esm_based'"
            )

    def adjust_education_level(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Maps education level scales to a unified standard across datasets.

        This method standardizes the education level data by mapping dataset-specific scales to a common
        scale, ensuring consistent meaning for each step across all samples. The unified scale is as follows:
            1: No education level
            2: Primary education level / Hauptschule
            3: Lower secondary education / Mittlere Reife
            4: A-level / Abitur oder Fachhochschulreife
            5: Degree from university or FH
            6: PhD / Promotion

        Args:
            df_traits: A pandas DataFrame containing the trait data.

        Returns:
            pd.DataFrame: A DataFrame with the education level column mapped to the unified scale.
        """
        education_cfg = self.config_parser(
            self.cfg_preprocessing["person_level"]["sociodemographics"],
            "continuous",
            "education_level",
        )[0]
        col = education_cfg["item_names"][self.dataset]

        if self.dataset in education_cfg["category_mappings"]:
            mapping = education_cfg["category_mappings"][self.dataset]
            df_traits[col] = df_traits[col].map(mapping)

        return df_traits

    def convert_str_cols_to_list_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts columns containing multi-number strings (e.g., "1, 2, 3") into lists of numbers.

        This method processes each column in the DataFrame and applies a conversion function to transform
        strings with comma-separated numbers into lists of integers. The changes are logged for columns
        where modifications occur.

        Args:
            df: Input DataFrame to process.

        Returns:
            pd.DataFrame: Processed DataFrame with lists of numbers replacing comma-separated strings.
        """
        df_copy = df.copy()

        for col in df.columns:
            df[col] = df[col].apply(lambda x: self._convert_str_to_list(x, col))

            if not df[col].equals(df_copy[col]):
                self.logger.log(f"-----Converted multi-number string to list for {col}")
        return df

    def _convert_str_to_list(
        self, cell_value: str, column_name: str
    ) -> list[int | str] | str:
        """
        Converts a string of comma-separated values into a list of integers or strings.

        This method processes cell values to identify strings with comma-separated numbers. It splits
        the string into individual elements, converting each to an integer if possible. If non-digit
        values are encountered, they remain as strings, and a log entry is made. If the input is not
        a valid string or does not match the expected pattern, the original value is returned.

        Args:
            cell_value: The value from the DataFrame cell to be processed.
            column_name: The name of the column being processed, used for logging.

        Returns:
            list[int | str] | str: A list of integers or strings if conversion is successful;
            otherwise, the original value.
        """
        pattern = r"^(\s*\d+\s*,\s*)*\d+\s*$"  # Number followed by comma

        if isinstance(cell_value, str) and re.match(pattern, cell_value):
            result = []

            for x in cell_value.split(","):
                x = x.strip()

                if x.isdigit():
                    result.append(int(x))

                else:
                    self.logger.log(
                        f"-----Found non-digit value '{x}' in column {column_name}"
                    )
                    result.append(x)
            return result

        else:
            return cell_value

    def dataset_specific_trait_processing(
        self, df_traits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Performs dataset-specific processing on the trait data.

        This method is designed to be overridden in subclasses for handling special use cases
        or dataset-specific processing requirements. If not specific processing is needed, the base
        implementation is used and the input DataFrame is returned unchanged.

        Args:
            df_traits: A pandas DataFrame containing the trait data to process.

        Returns:
            pd.DataFrame: The processed DataFrame, or the input DataFrame unchanged in the base class.
        """
        return df_traits

    def align_scales(
        self, df: pd.DataFrame, cat_list: list[str], df_type: str = "person_level"
    ) -> pd.DataFrame:
        """
        Aligns the scales of specified columns in the DataFrame according to the configuration.

        This method processes columns specified in the configuration for the given data type and
        feature categories (`cat_list`). For each column, it uses the scaling mappings defined in
        `align_scales_mapping` to transform the values to a unified scale. Only columns where both
        `item_names` and `align_scales_mapping` are defined for the current dataset are processed.

        Args:
            df: The DataFrame containing either trait or ESM data.
            df_type: The type of data being processed, either "person_level" or "esm_based".
            cat_list: A list of feature category strings to process.

        Returns:
            pd.DataFrame: The DataFrame with aligned scales.
        """
        specific_cfg = self.cfg_preprocessing[df_type]

        for cat in cat_list:
            for entry in specific_cfg[cat]:
                if "item_names" in entry:
                    item_names = entry["item_names"].get(self.dataset)
                    align_mapping = entry.get("align_scales_mapping", {}).get(
                        self.dataset
                    )

                    if item_names and align_mapping:
                        if isinstance(item_names, str):
                            item_names = [item_names]

                        old_min = min(align_mapping["min"].keys())
                        old_max = max(align_mapping["max"].keys())
                        new_min = align_mapping["min"][old_min]
                        new_max = align_mapping["max"][old_max]

                        for col in item_names:
                            self.logger.log(
                                f"        Execute align scales for column {col}"
                            )
                            if is_numeric_dtype(df[col]):
                                self.logger.log(
                                    f"          Old min of {col}: {df[col].min()}"
                                )
                                self.logger.log(
                                    f"          Old max of {col}: {df[col].max()}"
                                )
                            df[col] = df[col].apply(
                                lambda x: self._align_value(
                                    value=x,
                                    old_min=old_min,
                                    old_max=old_max,
                                    new_min=new_min,
                                    new_max=new_max,
                                )
                            )

                            self.logger.log(
                                f"        align scales executed for column {col}"
                            )
                            if is_numeric_dtype(df[col]):
                                self.logger.log(
                                    f"          New min of {col}: {df[col].min()}"
                                )
                                self.logger.log(
                                    f"          New max of {col}: {df[col].max()}"
                                )
        return df

    def _align_value(
        self,
        value: float,
        old_min: float,
        old_max: float,
        new_min: float,
        new_max: float,
    ) -> float:
        """
        Aligns a single value from an old scale range to a new scale range.

        This method scales a given value from the original range (`old_min` to `old_max`) to a new range
        (`new_min` to `new_max`). If the value is outside the original range, it is replaced with NaN to
        prevent bias in the transformation. The method also handles edge cases, such as values being NaN
        or the old range having equal minimum and maximum values.

        - NaN values are returned as is.
        - String inputs are coerced into numeric values; invalid strings are converted to NaN.
        - Values outside the original range are excluded, as this indicates non-valid values or an NaN category.
        - If `old_min` and `old_max` are equal, the value is set to `new_min` to avoid division by zero.

        Args:
            value: The value to be aligned.
            old_min: The minimum value of the original scale.
            old_max: The maximum value of the original scale.
            new_min: The minimum value of the new scale.
            new_max: The maximum value of the new scale.

        Returns:
            float: The value aligned to the new scale, or NaN if the input is invalid or out of range.
        """
        if pd.isna(value):
            return value

        if isinstance(value, str):
            value = pd.to_numeric(value, errors="coerce")

        if value < old_min or value > old_max:
            self.logger.log(
                f"         Value {value} out of range [{old_min}, {old_max}]. Setting to NaN before aligning the scales."
            )
            return np.nan

        if old_min == old_max:
            return new_min

        return new_min + ((value - old_min) * (new_max - new_min)) / (old_max - old_min)

    def check_scale_endpoints(self, df: pd.DataFrame, df_type: str) -> pd.DataFrame:
        """
        Validates that all values in the specified DataFrame columns fall within defined scale endpoints.

        This method checks if, after aligning scales, there are values outside the valid scale range specified
        in the configuration (`scale_endpoints`). Any values outside the range are set to `np.nan`, as they
        typically correspond to missing data or "I do not want to answer this" responses. Logs warnings for
        any out-of-range values detected.

        - Scale endpoints (`min` and `max`) are fetched from the configuration.
        - Column values are coerced to numeric types to ensure valid comparisons.
        - Values outside the scale range are replaced with `np.nan`.

        Args:
            df: The DataFrame to validate.
            df_type: The type of data being processed (e.g., "person_level" or "esm_based").

        Returns:
            pd.DataFrame: The DataFrame with out-of-range values replaced by `np.nan`.
        """
        for cat, cat_entries in self.cfg_preprocessing[df_type].items():
            for var in cat_entries:
                if "scale_endpoints" in var and self.dataset in var["item_names"]:
                    scale_min = var["scale_endpoints"]["min"]
                    scale_max = var["scale_endpoints"]["max"]
                    col_names = var["item_names"][self.dataset]

                    if isinstance(col_names, str):
                        col_names = [col_names]
                    for col_name in col_names:
                        column_values = pd.to_numeric(df[col_name], errors="coerce")
                        outside_values = column_values[
                            (column_values < scale_min) | (column_values > scale_max)
                        ]

                        if not outside_values.empty:
                            self.logger.log(
                                f"     WARNING: Values out of scale bounds in column '{col_name}': {outside_values.tolist()}, "
                                f"set to np.nan"
                            )
                            df[col_name] = df[col_name].where(
                                df[col_name].between(scale_min, scale_max), other=np.nan
                            )
        return df

    def create_binary_vars_from_categoricals(
        self, df_traits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Creates binary predictors from categorical variables in the DataFrame based on the configuration.

        This method processes categorical variables specified in the configuration and creates binary
        predictor columns. The configuration defines:
            - The dataset-specific column names to process (`item_names`).
            - The variable type (`var_type`) that must be "binary" to qualify.
            - The mapping of categories to binary values (`category_mappings`).

        For each variable:
        - If `item_names` is a string, it creates a single binary column based on the mapping.
        - If `item_names` is a list, it processes multiple columns to create the binary column.
        - Logs a warning and skips processing if `item_names` is neither a string nor a list.

        Args:
            df_traits: The DataFrame containing the trait data.

        Returns:
            pd.DataFrame: The DataFrame with new binary columns added.
        """
        demographic_cfg = self.cfg_preprocessing["person_level"]["sociodemographics"]

        for entry in demographic_cfg:
            if self.dataset in entry["item_names"] and entry["var_type"] == "binary":
                item_names = entry["item_names"][self.dataset]
                new_column_name = entry["name"]
                category_mappings = entry.get("category_mappings", {}).get(
                    self.dataset, {}
                )

                match item_names:
                    case str(column_name):
                        df_traits[new_column_name] = self._map_column_to_binary(
                            df=df_traits,
                            column=column_name,
                            mapping=category_mappings,
                            fill_na_with_zeros=False,
                        )
                        self.logger.log(
                            f"        Created {new_column_name} from {column_name}"
                        )

                    case list(columns):
                        df_traits[new_column_name] = self._map_columns_to_binary(
                            df=df_traits,
                            columns=columns,
                            mapping=category_mappings,
                            fill_na_with_zeros=False,
                        )
                        self.logger.log(
                            f"        Created {new_column_name} from {columns}"
                        )

                    case _:
                        self.logger.log(
                            f"        WARNING: {item_names} are neither of type str nor of type list, skip"
                        )
                        continue

        return df_traits

    def _map_column_to_binary(
        self,
        df: pd.DataFrame,
        column: str,
        mapping: dict[int, int],
        fill_na_with_zeros: bool = False,
    ) -> pd.Series:
        """
        Maps a single column to a binary column based on the provided mapping.

        This method transforms a categorical column into a binary column using the specified mapping.
        It supports two types of column data:
        - **Object (e.g., strings with comma-separated values)**: Each cell is processed to match any
          value in the mapping, considering multiple values separated by commas.
        - **Non-object (e.g., numeric)**: The mapping is applied directly to each value.

        If the input contains ambiguous or missing values, their handling is controlled by:
        - **`map_ambiguous`** (always `True` here for comma-separated values).
        - **`fill_na_with_zeros`**: If `True`, missing values are replaced with zeros in the binary output.

        Args:
            df: The DataFrame containing the data.
            column: The name of the column to transform into a binary column.
            mapping: A dictionary specifying the mapping from categorical values to binary.
            fill_na_with_zeros: Whether to fill missing values with zeros in the resulting binary column.

        Returns:
            pd.Series: A binary column derived from the specified categorical column.
        """
        if df[column].dtype == object:
            return df[column].apply(
                lambda x: self._map_comma_separated(
                    cell_value=x,
                    mapping=mapping,
                    map_ambiguous=True,
                    fill_na_with_zeros=fill_na_with_zeros,
                )
            )

        return df[column].map(
            lambda x: self._map_single_value(
                cell_value=x, mapping=mapping, fill_na_with_zeros=fill_na_with_zeros
            )
        )

    def _map_comma_separated(
        self,
        cell_value: str,
        mapping: dict[int, int],
        map_ambiguous: bool = True,
        fill_na_with_zeros: bool = False,
    ) -> Union[int, float]:
        """
        Processes cells containing single or comma-separated values and maps them to binary or numeric outputs.

        This method handles categorical data that may contain multiple values separated by commas. Each value is
        processed individually using the provided mapping. If `map_ambiguous` is `True`, cells with mixed mappings
        (e.g., some values map to `1` and others to `0`) are handled as follows:
        - If `map_ambiguous` is `False`, ambiguous values are set to `np.nan`.
        - Otherwise, the highest mapped value is returned.

        Missing or invalid values are processed based on `fill_na_with_zeros`:
        - If `fill_na_with_zeros` is `True`, missing values are mapped to `0`.
        - Otherwise, missing values are set to `np.nan`.

        Args:
            cell_value: A string containing a single or comma-separated values to be processed.
            mapping: A dictionary specifying the mapping for individual values.
            map_ambiguous: Whether to process ambiguous mappings. Defaults to `True`.
            fill_na_with_zeros: Whether to replace missing values with zeros. Defaults to `False`.

        Returns:
            Union[int, float]: The mapped value (e.g., `0`, `1`, or `np.nan` for invalid or ambiguous cases).
        """
        if pd.isna(cell_value):
            if fill_na_with_zeros:
                return 0
            else:
                return np.nan

        values = cell_value.split(",")
        mapped_values = [
            self._map_single_value(int(val.strip()), mapping)
            for val in values
            if val.strip().isdigit()
        ]

        if not map_ambiguous:
            if len(set(mapped_values)) > 1:
                mapped_values = [np.nan]

        return max(mapped_values)

    @staticmethod
    def _map_single_value(
        cell_value, mapping: dict[int, int], fill_na_with_zeros: bool = False
    ) -> Union[int, float]:
        """
        Maps a single value using the provided mapping dictionary.

        This method applies a mapping to a single value, handling missing or unmapped values as follows:
        - If the input value is `np.nan`, it is treated as missing:
            - If `fill_na_with_zeros` is `True`, it is mapped to `0`.
            - Otherwise, it remains as `np.nan`.
        - If the input value is not found in the mapping, it is mapped to `0`.

        Args:
            cell_value: The value to map. Can be numeric, a string, or `None` (including `np.nan`).
            mapping: A dictionary specifying the mapping for valid values.
            fill_na_with_zeros: Whether to map missing (`np.nan`) values to `0`. Defaults to `False`.

        Returns:
            Union[int, float]: The mapped value or `np.nan` for missing values.
        """
        if pd.isna(cell_value):
            if fill_na_with_zeros:
                return 0
            else:
                return np.nan

        return mapping.get(cell_value, 0)

    def _map_columns_to_binary(
        self,
        df: pd.DataFrame,
        columns: list[str],
        mapping: dict[int, int],
        fill_na_with_zeros: bool = False,
    ) -> pd.Series:
        """
        Maps multiple columns to a single binary column based on the provided mapping.

        This method combines the values from multiple columns into a single binary column by applying
        a mapping to each column. The binary output for each row is determined by taking the maximum
        binary value across all specified columns for that row.

        Args:
            df: The DataFrame containing the data.
            columns: A list of column names to process.
            mapping: A dictionary specifying the mapping for categorical values.
            fill_na_with_zeros: Whether to replace missing values with zeros in the binary output. Defaults to `False`.

        Returns:
            pd.Series: A binary column derived from the specified categorical columns.
        """
        return df.apply(
            lambda row: max(
                [
                    self._map_column_to_binary(
                        pd.DataFrame({col: [row[col]]}),
                        col,
                        mapping,
                        fill_na_with_zeros,
                    ).iloc[0]
                    for col in columns
                ]
            ),
            axis=1,
        )

    @abstractmethod
    def merge_states(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges state data files into a single DataFrame for the specified dataset.

        This method combines multiple DataFrames containing state data into a single DataFrame.
        The files to merge are specified in `self.path_to_raw_data`. Subclasses must implement
        this method to handle dataset-specific merging logic.

        Args:
            df_dct: A dictionary where the keys are filenames and the values are pandas DataFrames
                    corresponding to the loaded data files.

        Returns:
            pd.DataFrame: A single DataFrame containing all state variables for the dataset.
        """
        pass

    def dataset_specific_state_processing(
        self, df_states: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Applies dataset-specific processing to state data.

        This method is designed to be overridden in subclasses to handle special use cases or
        dataset-specific processing requirements for state data. The default implementation
        returns the input DataFrame unchanged.

        Args:
            df_states: A pandas DataFrame containing state data to be processed.

        Returns:
            pd.DataFrame: The processed DataFrame, or the input DataFrame unchanged if not overridden.
        """
        return df_states

    def create_person_level_vars_from_esm(
        self, df_states: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregates ESM data to create person-level variables.

        This method serves as a wrapper for various steps that transform state-level ESM data into
        person-level variables and aggregates the data.

        Steps performed:
        - Creating descriptive statistics for srmc variables (e.g., sleep_quality).
        - Calculating percentages of categorical variables (e.g., `ftf_interactions`).
        - Calculating variables that need custom processing
            - Calculating the number of specific types of interactions.
            - Deriving weekday response patterns.
            - Deriving early-day response patterns.
            - Calculating the number of responses.
            - Calculating percentages of responses.
        - Create a column with the year(s) of participation as we need this for the country-level data
        - Averaging criterion items.

        Args:
            df_states: A pandas DataFrame containing the state-level ESM data.

        Returns:
            pd.DataFrame: The processed DataFrame aggregated at the person level.
        """
        steps = [
            (
                self.create_person_level_desc_stats,
                {"df": None, "feature_category": "self_reported_micro_context"},
            ),
            (self.create_person_level_percentages, {"df_states": None}),
            (self.create_number_interactions, {"df_states": None}),
            (self.create_weekday_responses, {"df_states": None}),
            (self.create_early_day_responses, {"df_states": None}),
            (self.create_number_responses, {"df_states": None}),
            (self.create_percentage_responses, {"df_states": None}),
            (self.create_years_of_participation, {"df_states": None}),
            (self.average_criterion_items, {"df_states": None}),
        ]

        for method, kwargs in steps:
            kwargs = {k: v if v is not None else df_states for k, v in kwargs.items()}
            df_states = self._log_and_execute(method, indent=6, **kwargs)

        return df_states

    def create_person_level_desc_stats(
        self, df: pd.DataFrame, feature_category: str
    ) -> pd.DataFrame:
        """
        Calculates descriptive statistics (mean, standard deviation, min, max) for continuous variables
        from ESM or sensing-based data, and includes them as person-level predictors.

        This method processes continuous variables based on the specified `feature_category`.
        - For `self_reported_micro_context` (ESM data), variables such as `number_interaction_partners`
          and `sleep_quality` are processed.
        - For `sensing_based` data, variables from phone and GPS/weather data are included.

        **Key Details**:
        - Requires at least two non-NaN values per person to calculate standard deviation, min, and max.
        - Groups the data by person ID (`raw_esm_id_col` or `raw_sensing_id_col`) to compute statistics.
        - Filters out groups with fewer than two valid values before calculating statistics.
        - The original variable columns are dropped after the aggregated statistics are added to the DataFrame.

        Args:
            df: The DataFrame containing state-level data.
            feature_category: The type of data to process, either:
                - "self_reported_micro_context" for ESM data, or
                - "sensing_based" for sensing data.

        Returns:
            pd.DataFrame: The updated DataFrame with descriptive statistics for continuous variables.

        Raises:
            ValueError: If `feature_category` is not "self_reported_micro_context" or "sensing_based".
            KeyError: If an expected column for a variable is missing in the DataFrame.
        """
        if feature_category == "self_reported_micro_context":
            cont_var_entries = self.config_parser(
                self.cfg_preprocessing["esm_based"]["self_reported_micro_context"],
                "continuous",
                "number_interaction_partners",
                "sleep_quality",
            )
            id_col = self.raw_esm_id_col

        elif feature_category == "sensing_based":
            phone_entries = self.config_parser(
                self.cfg_preprocessing["sensing_based"]["phone"], "continuous"
            )
            gps_weather_entries = self.config_parser(
                self.cfg_preprocessing["sensing_based"]["gps_weather"], "continuous"
            )
            cont_var_entries = phone_entries + gps_weather_entries
            id_col = self.raw_sensing_id_col

        else:
            raise ValueError(
                "Feature category must be 'self_reported_micro_context' or 'sensing_based'"
            )

        grouped_df = df.groupby(id_col)
        for entry in cont_var_entries:
            var_name = entry["name"]
            if self.dataset in entry["item_names"]:
                column = entry["item_names"][self.dataset]

                if column in df.columns:
                    valid_counts = grouped_df[column].count().reset_index(name="count")
                    valid_groups = valid_counts[valid_counts["count"] >= 2][id_col]
                    filtered_df = df[df[id_col].isin(valid_groups)]
                    grouped_filtered_df = filtered_df.groupby(id_col)

                    stats = (
                        grouped_filtered_df[column]
                        .agg(
                            mean="mean",
                            sd=lambda x: x.std() if x.count() >= 2 else pd.NA,
                            min=lambda x: x.min() if x.count() >= 2 else pd.NA,
                            max=lambda x: x.max() if x.count() >= 2 else pd.NA,
                        )
                        .reset_index()
                    )

                    stats.columns = [
                        id_col,
                        f"{var_name}_mean",
                        f"{var_name}_sd",
                        f"{var_name}_min",
                        f"{var_name}_max",
                    ]

                    self.logger.log(
                        f"          Created M, SD, Min, and Max for var {var_name}"
                    )

                    df = pd.merge(df, stats, on=id_col, how="left")
                    df = df.drop(column, axis=1)

                else:
                    raise KeyError(
                        f"Column: {column} not found in {self.dataset} state_df"
                    )

        return df

    def create_person_level_percentages(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Creates person-level percentages for specified ESM-based variables.

        This method calculates person-level ratios based on dataset-specific columns defined in the configuration.
        For each variable, it processes the data, applies category mappings if needed, and calculates the percentage
        of valid responses grouped by person. The results are added as new columns to the original DataFrame.

        **Implementation Logic**:
        1. Extract configuration entries with `var_type="percentage"`.
        2. For each variable:
           - Apply category mappings for binary classification if defined in the configuration.
           - For variables marked as "per_day", group data by person and day.
           - Calculate the percentage of `1`s relative to valid responses (`0` or `1`) for each person.
           - Validate that calculated percentages do not exceed `1`, crop if necessary. (e.g., 1.05 is set to 1)
        3. Merge the calculated percentages into the original DataFrame.

        Args:
            df_states: The DataFrame containing state-level ESM data.

        Returns:
            pd.DataFrame: The original DataFrame with new person-level percentage columns added.

        Raises:
            KeyError: If a required column for a variable is not found in the dataset.
            ValueError: If the datatype of a column is invalid for processing.
        """
        percentage_var_entries = self.config_parser(
            cfg=self.cfg_preprocessing["esm_based"]["self_reported_micro_context"],
            var_type="percentage",
        )
        person_level_stats = []

        for entry in percentage_var_entries:
            var_name = entry["name"]
            if "item_names" in entry and self.dataset in entry["item_names"]:
                column = entry["item_names"][self.dataset]

                if isinstance(column, str):
                    if column in df_states.columns:
                        df_copied = deepcopy(df_states)

                        if (
                            "category_mappings" in entry
                            and self.dataset in entry["category_mappings"]
                        ):
                            category_mapping = entry["category_mappings"][self.dataset]

                            if df_copied[column].dtype in [object, str]:
                                df_copied[column] = df_copied[column].apply(
                                    lambda x: self._map_comma_separated(
                                        x, category_mapping, False
                                    )
                                )

                            elif df_copied[column].dtype in [int, float]:
                                df_copied[column] = df_copied[column].apply(
                                    lambda x: self._map_single_value(
                                        x, category_mapping
                                    )
                                )

                            else:
                                raise ValueError(
                                    f"{column} dtype must be object, str, int, or float"
                                )

                        if "per_day" in entry:
                            df_copied[self.esm_timestamp] = pd.to_datetime(
                                df_copied[self.esm_timestamp], errors="coerce"
                            ).dt.date

                        df_copied = (
                            df_copied[[self.raw_esm_id_col, self.esm_timestamp, column]]
                            .drop_duplicates()
                            .sort_values([self.raw_esm_id_col, self.esm_timestamp])
                        )
                        grouped_df = df_copied.groupby(self.raw_esm_id_col)

                        # If no valid values are found np.nan is assigned (this also results in RuntimeWarning)
                        stats = (
                            grouped_df[column]
                            .apply(
                                lambda group: (group == 1).sum()
                                / group.isin([0, 1]).sum()
                            )
                            .reset_index()
                        )
                        stats.columns = [self.raw_esm_id_col, f"{var_name}"]
                        person_level_stats.append(stats)
                        self.logger.log(
                            f"          Created Percentage for var {var_name}"
                        )
                        assert (
                            len(stats[stats[var_name] > 1]) == 0
                        ), "percentage found that is greater than 1"

                    else:
                        raise KeyError(
                            f"Column: {column} not found in {self.dataset} state_df"
                        )

                else:
                    self.logger.log(
                        f"      Skipping var {var_name} in dataset {self.dataset}, has custom method"
                    )
                    continue

        if person_level_stats:
            for stats_df in person_level_stats:
                df_states = pd.merge(
                    df_states, stats_df, on=self.raw_esm_id_col, how="left"
                )

        return df_states

    def create_number_interactions(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the person-level variable `number_interactions`.

        This method calculates the total number of social interactions for each person by summing all rows
        where they indicated a social interaction. The columns and category mappings for each dataset are
        specified in the configuration.

        Note:
        - This method is overridden in the CoCoMS dataset for a custom implementation.

        Args:
            df_states: A pandas DataFrame containing state-level ESM data.

        Returns:
            pd.DataFrame: The original DataFrame with a new column, `number_interactions`, added.
        """
        cfg_num_ia = self.config_parser(
            self.cfg_preprocessing["esm_based"]["self_reported_micro_context"],
            "continuous",
            "number_interactions",
        )[0]
        item_name = cfg_num_ia["item_names"][self.dataset]
        cat_mappings = cfg_num_ia["category_mappings"][self.dataset]

        df_states["interaction_sum"] = df_states[item_name].map(
            lambda x: cat_mappings.get(x, 0)
        )
        df_grouped = df_states.groupby(self.raw_esm_id_col, as_index=False).agg(
            number_interactions=("interaction_sum", "sum")
        )
        df_states = df_states.merge(df_grouped, on=self.raw_esm_id_col, how="left")

        return df_states

    def create_weekday_responses(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the percentage of responses that occurred on weekdays (Monday to Friday) for each person.

        This method groups responses by person ID (`self.raw_esm_id_col`) and calculates the percentage of
        responses that took place on weekdays. The result is added to the DataFrame as a new column,
        `weekday_responses`.

        Args:
            df_states: A pandas DataFrame containing response data and timestamp information.

        Returns:
            pd.DataFrame: The modified DataFrame with an added column, `weekday_responses`.

        Raises:
            AssertionError: If any calculated percentage exceeds 1, indicating a logic error.
        """
        df_states[self.esm_timestamp] = pd.to_datetime(
            df_states[self.esm_timestamp], errors="coerce"
        )
        df_states["is_weekday"] = (
            df_states[self.esm_timestamp].dt.weekday < 5
        )  # True for Monday to Friday

        weekday_stats = df_states.groupby(self.raw_esm_id_col)["is_weekday"].mean()
        df_states = df_states.merge(
            weekday_stats.rename("weekday_responses"),
            on=self.raw_esm_id_col,
            how="left",
        )
        assert (
            len(df_states[df_states["weekday_responses"] > 1]) == 0
        ), "percentage found that is greater than 1"

        return df_states

    def create_early_day_responses(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the percentage of responses that occurred in the 'early' time window (3 AM to 3 PM).

        This method adjusts timestamps to use a daily cutoff at 3 AM, identifies responses that occurred
        between 3 AM and 3 PM, and calculates the percentage of such responses for each person, grouped by
        their ID (`self.raw_esm_id_col`). The result is added to the DataFrame as a new column,
        `early_day_responses`.

        Args:
            df_states: A pandas DataFrame containing response data and timestamp information.

        Returns:
            pd.DataFrame: The modified DataFrame with an added column, `early_day_responses`.

        Raises:
            AssertionError: If any calculated percentage exceeds 1, indicating a logic error.
        """
        df_states[self.esm_timestamp] = pd.to_datetime(
            df_states[self.esm_timestamp], errors="coerce"
        )
        df_states["hour_adjusted"] = (
            df_states[self.esm_timestamp] - pd.DateOffset(hours=3)
        ).dt.hour
        df_states["is_early"] = df_states["hour_adjusted"].between(
            0, 11, inclusive="both"
        )

        early_stats = df_states.groupby(self.raw_esm_id_col)["is_early"].mean()
        df_states = df_states.merge(
            early_stats.rename("early_day_responses"),
            on=self.raw_esm_id_col,
            how="left",
        )
        assert (
            len(df_states[df_states["early_day_responses"] > 1]) == 0
        ), "percentage found that is greater than 1"

        return df_states

    def create_number_responses(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the total number of responses per person.

        This method counts the total number of responses for each person, where each row represents one
        response, and adds the result as a new column, `number_responses`, to the DataFrame.

        Args:
            df_states: A pandas DataFrame containing response data.

        Returns:
            pd.DataFrame: The modified DataFrame with an added column, `number_responses`, per person.
        """
        response_counts = (
            df_states.groupby(self.raw_esm_id_col)
            .size()
            .reset_index(name="number_responses")
        )
        df_states = df_states.merge(response_counts, on=self.raw_esm_id_col, how="left")

        return df_states

    def create_percentage_responses(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the percentage of responses per person.

        This method computes the percentage of responses for each person by dividing the actual number of responses
        (`number_responses`) by the maximum possible responses (`max_responses`) for the dataset. The method
        adjusts the calculation for datasets where participants may have participated in multiple study waves.

        **Implementation Details**:
        - `max_responses` is fetched from the configuration and represents the maximum possible responses for the dataset.
        - For datasets like `emotions` and `zpid`, participants in two bursts are identified using the `studyWave` column
          (from the configuration). The `max_responses` is doubled for such participants.
        - For other datasets, no adjustments are made, and the percentage is calculated directly as
          `number_responses / max_responses`.
        - Percentages greater than `1` are logged as warnings and capped at `1` to handle potential anomalies.

        Args:
            df_states: A pandas DataFrame containing the `number_responses` column.

        Returns:
            pd.DataFrame: The modified DataFrame with an added column, `percentage_responses`.

        Raises:
            KeyError: If required columns or mappings are missing from the configuration.
        """
        max_responses = self.config_parser(
            self.cfg_preprocessing["esm_based"]["self_reported_micro_context"],
            "percentage",
            "percentage_responses",
        )[0]["special_mappings"][self.dataset]
        study_wave_col = self.config_parser(
            self.cfg_preprocessing["esm_based"]["other_esm_columns"],
            "string",
            "studyWave",
        )[0]["item_names"]

        if self.dataset in [
            "emotions",
            "zpid",
        ]:  # need to adjust these datasets for two bursts
            df_states["percentage_responses"] = df_states.apply(
                lambda row: (
                    row["number_responses"] / (max_responses * 2)
                    if row[study_wave_col[self.dataset]] == "Both"
                    else row["number_responses"] / max_responses
                ),
                axis=1,
            )
        else:
            df_states["percentage_responses"] = (
                df_states["number_responses"] / max_responses
            )

        max_crit_resp = df_states[df_states["percentage_responses"] > 1]
        if len(max_crit_resp) > 0:
            self.logger.log(
                f"        WARNING: Found values over 1, Max value: {max_crit_resp['percentage_responses'].max()}, set to 1"
            )

        df_states["percentage_responses"] = df_states["percentage_responses"].apply(
            lambda x: 1 if x > 1 else x
        )

        return df_states

    def create_years_of_participation(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a column indicating the year(s) in which participants answered ESM surveys.

        Each participant's years of participation are represented as a tuple in the new column
        `years_of_participation`. If no valid data is available, the years of data collection
        specified in the configuration are used.

        Args:
            df_states: A pandas DataFrame containing response data and timestamp information.

        Returns:
            pd.DataFrame: The DataFrame with an additional column `years_of_participation`.
        """
        df_states["year"] = pd.to_datetime(df_states[self.esm_timestamp]).dt.year
        df_years = (
            df_states.groupby(self.raw_esm_id_col)["year"]
            .apply(lambda x: tuple(sorted(x.unique())))
            .reset_index()
        )
        df_years = df_years.rename(columns={"year": "years_of_participation"})

        df_states = df_states.merge(df_years, on=self.raw_esm_id_col, how="left")
        df_states = df_states.drop(columns=["year"])
        years_of_data_collection = tuple(
            self.cfg_preprocessing["general"]["years_of_data_collection"][self.dataset]
        )

        def _contains_nan(value: Union[tuple, Any]) -> bool:
            """
            Checks if a value or any element in a tuple contains NaN.

            Args:
                value: The value to check. It can be a single value or a tuple of values.

            Returns:
                bool: True if the value or any element in the tuple is NaN, False otherwise.
            """
            if isinstance(value, tuple):
                return any(pd.isna(v) for v in value)
            return pd.isna(value)

        df_states["years_of_participation"] = df_states["years_of_participation"].apply(
            lambda x: years_of_data_collection if _contains_nan(x) else x
        )

        return df_states

    def average_criterion_items(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Computes person-level averages for items assessing positive and negative state affect.

        This method calculates the average scores for positive and negative affect items based on
        the configuration and assigns these averages back to the original columns for each person.

        Args:
            df_states: A pandas DataFrame containing state-level data.

        Returns:
            pd.DataFrame: The modified DataFrame with averaged scores for positive and negative affect items.
        """
        affect_states_dct = self._get_state_affect_dct()

        for affect_state, affect_cols in affect_states_dct.items():
            if affect_cols:
                person_avg = df_states.groupby(self.raw_esm_id_col)[
                    affect_cols
                ].transform("mean")
                df_states[affect_cols] = person_avg

        return df_states

    def _get_state_affect_dct(self) -> dict[str, list[str]]:
        """
        Retrieves a dictionary of affect types (positive and negative) and their corresponding items for the dataset.

        This helper function extracts the criterion variables for positive affect (PA) and negative affect (NA)
        from the configuration and returns a dictionary where:
        - Keys represent the type of affect (e.g., "pa_state", "na_state").
        - Values are lists of item names specific to the current dataset.

        Returns:
            dict[str, list[str]]: A dictionary mapping affect types to their respective item names for the dataset.
        """
        affect_var_entries = self.config_parser(
            self.cfg_preprocessing["esm_based"]["criterion"],
            "continuous",
            "pa_state",
            "na_state",
        )
        affect_states_dct = {
            val["name"]: val["item_names"][self.dataset]
            for val in affect_var_entries
            if self.dataset in val["item_names"]
        }

        return affect_states_dct

    def filter_min_num_esm_measures(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Filters participants based on the minimum number of valid ESM measurements.

        This method retains only participants who provided at least a specified number of valid ESM measurements,
        as defined in the configuration (`min_num_esm_measures`). Valid measurements are determined based on
        the presence of non-missing data in the momentary well-being indicators (positive and negative affect items).
        In the analysis for the paper, min_num_esm_measures is five.

        For datasets with study waves (i.e., "cocout", "cocoms"), the method logs additional information
        about the filtering process for each wave.

        Args:
            df_states: A pandas DataFrame containing state-level ESM data.

        Returns:
            pd.DataFrame: The filtered DataFrame containing only participants meeting the minimum measurement requirement.

        Raises:
            ValueError: If no positive or negative affect items are found in the configuration for the dataset.
        """
        affect_states_dct = self._get_state_affect_dct()
        min_num_esm = self.cfg_preprocessing["general"]["min_num_esm_measures"]
        pos_affect_cols = affect_states_dct.get("pa_state", [])
        neg_affect_cols = affect_states_dct.get("na_state", [])

        if pos_affect_cols and neg_affect_cols:
            valid_measurements = df_states[pos_affect_cols].notna().any(
                axis=1
            ) & df_states[neg_affect_cols].notna().any(axis=1)
        elif pos_affect_cols:
            valid_measurements = df_states[pos_affect_cols].notna().any(axis=1)
        elif neg_affect_cols:
            valid_measurements = df_states[neg_affect_cols].notna().any(axis=1)
        else:
            raise ValueError(
                f"No positive or negative items found in config for {self.dataset}"
            )

        valid_count_per_person = (
            df_states[valid_measurements].groupby(self.raw_esm_id_col).size()
        )
        filtered_df = df_states[
            df_states[self.raw_esm_id_col].isin(
                valid_count_per_person[valid_count_per_person >= min_num_esm].index
            )
        ]
        persons_in_unfiltered_df = df_states[self.raw_esm_id_col].nunique()
        persons_in_filtered_df = filtered_df[self.raw_esm_id_col].nunique()

        self.logger.log(
            f"        N persons included in before filtering: {persons_in_unfiltered_df}"
        )
        self.logger.log(
            f"        N measurements included in before filtering: {len(df_states)}"
        )
        self.logger.log(
            f"        N persons after require at least {min_num_esm} measurements per person: {persons_in_filtered_df}"
        )
        self.logger.log(
            f"        N measurements after require at least {min_num_esm} measurements per person: {len(filtered_df)}"
        )

        if self.dataset in ["cocout", "cocoms"]:
            for wave in filtered_df["studyWave"].unique():
                df_filtered_tmp = filtered_df[filtered_df["studyWave"] == wave]
                df_unfiltered_tmp = df_states[df_states["studyWave"] == wave]

                self.logger.log(
                    f"        Split up filtered num measurements included for {self.dataset}"
                )
                self.logger.log(
                    f"          N persons for wave {wave} before filtering: {df_unfiltered_tmp[self.raw_esm_id_col].nunique()}"
                )
                self.logger.log(
                    f"          N measurements for wave {wave} before filtering: {len(df_unfiltered_tmp)}"
                )
                self.logger.log(
                    f"          N persons for wave {wave} after filtering: {df_filtered_tmp[self.raw_esm_id_col].nunique()}"
                )
                self.logger.log(
                    f"          N measurements for wave {wave} after filtering: {len(df_filtered_tmp)}"
                )
        return filtered_df

    def store_wb_items(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Stores well-being (wb) items from the state DataFrame for later postprocessing.

        This method extracts wb-related items from the state DataFrame, computes person-level scores
        (positive affect, negative affect, and well-being), and saves the resulting DataFrame to a file.
        The original state DataFrame is returned unchanged to maintain compatibility with the preprocessing logic.

        Key Details:
        - For the `zpid` dataset, the wb score is derived directly from the valence item.
        - For other datasets:
            - Positive affect (`pa_state`) and negative affect (`na_state`) scores are computed as the mean
              of their respective items.
            - The well-being score (`wb_state`) is calculated as the mean of the positive affect score and
              an inversely coded negative affect score.
        - The resulting DataFrame is saved as a pickle file in the preprocessing path specified in the configuration.

        Args:
            df_states: A pandas DataFrame containing state-level ESM data.

        Returns:
            pd.DataFrame: The original state DataFrame, unchanged.
        """
        cols = [self.raw_esm_id_col, self.esm_timestamp, "country"]

        affect_states_dct = self._get_state_affect_dct()
        cols.extend(
            [item for sublist in affect_states_dct.values() for item in sublist]
        )
        df_wb_items = df_states[cols]
        pa_items = affect_states_dct["pa_state"]

        if self.dataset == "zpid":  # only includes 'valence'
            df_wb_items["wb_state"] = df_wb_items[pa_items[0]]

        else:
            na_items = affect_states_dct["na_state"]
            df_wb_items[f"pa_state"] = df_wb_items[pa_items].mean(axis=1)
            df_wb_items[f"na_state"] = df_wb_items[na_items].mean(axis=1)

            df_wb_items[f"state_na_inv"] = inverse_code(
                df_wb_items[na_items], min_scale=1, max_scale=6
            ).mean(axis=1)
            new_columns = pd.DataFrame(
                {f"wb_state": df_wb_items[[f"pa_state", f"state_na_inv"]].mean(axis=1)}
            )
            df_wb_items = pd.concat([df_wb_items, new_columns], axis=1)
            df_wb_items = df_wb_items.drop(["state_na_inv"], axis=1)

        if self.cfg_preprocessing["general"]["store_wb_items"]:
            filename = os.path.join(
                self.cfg_preprocessing["general"]["path_to_preprocessed_data"],
                f"wb_items_{self.dataset}",
            )
            with open(filename, "wb") as f:
                pickle.dump(df_wb_items, f)

        return df_states

    def set_full_col_df_as_attr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Saves the full state DataFrame as a class attribute for later reference.

        This method creates a deep copy of the state DataFrame and assigns it to a class attribute
        (`df_before_final_selection`) to facilitate later sanity checks, such as calculating scale reliabilities.

        Args:
            df: A pandas DataFrame representing the state-level data.

        Returns:
            pd.DataFrame: The original DataFrame, unchanged.
        """
        self.df_before_final_selection = deepcopy(df)
        return df

    def dataset_specific_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies dataset-specific post-processing logic.

        This method is intended to be overridden in subclasses to implement post-processing
        logic specific to a particular dataset. The default implementation returns the input
        DataFrame unchanged.

        Args:
            df: A pandas DataFrame to be post-processed.

        Returns:
            pd.DataFrame: The processed DataFrame, or the input DataFrame if not overridden.
        """
        return df

    def collapse_df(self, df: pd.DataFrame, df_type: str) -> pd.DataFrame:
        """
        Collapses a state-level DataFrame into a person-level DataFrame.

        This method transforms a DataFrame that varies by person and ESM measurement or date into a person-level
        DataFrame with one row per person. It removes variables that vary within-person by retaining only columns
        with constant values per person.

        Args:
            df: A pandas DataFrame to be collapsed.
            df_type: The type of data, either "esm_based" or "sensing_based", used to determine the ID column.

        Returns:
            pd.DataFrame: A person-level DataFrame containing only the constant columns aggregated per person.

        Raises:
            ValueError: If `df_type` is not "esm_based" or "sensing_based".
        """
        if df_type == "esm_based":
            id_col = self.raw_esm_id_col
        elif df_type == "sensing_based":
            id_col = self.raw_sensing_id_col
        else:
            raise ValueError("df_type must be esm_based or sensing_based")

        constant_cols = []
        for col in df.columns:
            if df.groupby(id_col)[col].nunique().max() == 1:
                constant_cols.append(col)

        df_person_level = df[constant_cols].drop_duplicates(subset=id_col)

        return df_person_level

    def set_id_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns a unique ID to each row and sets it as the DataFrame's index.

        This method creates a new ID column in the format `self.dataset_<integer>` to uniquely identify
        each row in the DataFrame. The new ID column is then set as the index of the DataFrame.

        Args:
            df: A pandas DataFrame for which a new index is to be set.

        Returns:
            pd.DataFrame: The DataFrame with a new unique index.
        """
        df["new_id"] = self.dataset + "_" + (df.reset_index().index + 1).astype(str)
        df = df.set_index("new_id", drop=True)

        return df

    def extract_columns(
        self, config_data: Union[dict, NestedDict, list, str]
    ) -> list[str]:
        """
        Recursively extracts column names specific to the dataset from nested configuration data.

        This method traverses nested dictionaries, lists, or strings in the configuration data
        and collects column names that match the current dataset.

        Args:
            config_data: The configuration data, which may include nested dictionaries, lists, or strings.

        Returns:
            list[str]: A list of column names relevant to the current dataset.
        """
        columns = []
        for key, value in config_data.items():
            if isinstance(value, dict):
                columns.extend(self.extract_columns(value))

            elif isinstance(value, list) and key == self.dataset:
                columns.extend(value)

            elif isinstance(value, str) and key == self.dataset:
                columns.append(value)

        return columns

    def merge_country_data(
        self, country_var_dct: dict[str, pd.DataFrame], df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges country-level variables into participant-level data.

        This method:
        1. Merges multiple country-level DataFrames (e.g., health, psycho-political, socio-economic) on
           `country` and `year`.
        2. Explodes the `years_of_participation` column in the participant-level DataFrame to match
           individual years with the corresponding country-level data.
        3. Computes the average of country-level variables for participants whose assessments span multiple years.
        4. Merges the averaged country-level variables back into the participant-level DataFrame.

        Args:
            country_var_dct: A dictionary containing country-level DataFrames keyed by variable type.
            df: A participant-level DataFrame with a `years_of_participation` column (list of years) and
                a `country` column.

        Returns:
            pd.DataFrame: The participant-level DataFrame enriched with country-level variables.
        """
        df_country_level = reduce(
            lambda left, right: pd.merge(
                left, right, on=["country", "year"], how="outer"
            ),
            country_var_dct.values(),
        )
        df_country_level["democracy_index"] = pd.to_numeric(
            df_country_level["democracy_index"].str.replace(",", ".", regex=False),
            errors="coerce",
        )
        country_level_cols_to_agg = df_country_level.columns.drop(["country", "year"])

        df = df.reset_index()
        df["original_index"] = df.index

        df_exploded = df.explode("years_of_participation").rename(
            columns={"years_of_participation": "year"}
        )
        df_merged = pd.merge(
            df_exploded, df_country_level, on=["country", "year"], how="left"
        )
        mean_country_vars = (
            df_merged.groupby(self.raw_esm_id_col)[country_level_cols_to_agg]
            .mean()
            .reset_index()
        )

        df = pd.merge(df, mean_country_vars, on=self.raw_esm_id_col, how="left")
        df = df.drop(columns=["original_index"])

        return df

    def merge_dfs_on_id(
        self, df_states: pd.DataFrame, df_traits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges state-level and trait-level DataFrames on participant IDs.

        This method merges the `df_states` and `df_traits` DataFrames using their respective participant
        ID columns (`self.raw_esm_id_col` and `self.raw_trait_id_col`). The merge is performed as a
        left join, ensuring all rows in `df_states` are retained.

        Args:
            df_states: A pandas DataFrame containing state-level data.
            df_traits: A pandas DataFrame containing trait-level data.

        Returns:
            pd.DataFrame: A merged DataFrame combining state-level and trait-level data.
        """
        df_traits = df_traits.drop(columns=["country"])
        df_joint = pd.merge(
            df_states,
            df_traits,
            left_on=self.raw_esm_id_col,
            right_on=self.raw_trait_id_col,
            how="left",
        )
        return df_joint

    def inverse_coding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs inverse coding on specific items in the DataFrame based on configuration.

        This method recodes items for the current dataset (`self.dataset`) using the formula:
        `new_value = scale_max + scale_min - old_value`. The items to be recoded and their scale endpoints
        (`min` and `max`) are defined in the configuration. It is assumed that the scales have already
        been aligned before this step.

        Args:
            df: A pandas DataFrame containing the items to be recoded.

        Returns:
            pd.DataFrame: The DataFrame with recoded items where applicable.

        Raises:
            KeyError: If a specified item to be recoded is not found in the DataFrame.
        """
        cont_pers_entries = self.config_parser(
            self.cfg_preprocessing["person_level"]["personality"], "continuous"
        )
        for entry in cont_pers_entries:
            if "items_to_recode" in entry and entry["items_to_recode"].get(
                self.dataset
            ):
                items_to_recode = entry["items_to_recode"][self.dataset]
                scale_min = entry["scale_endpoints"]["min"]
                scale_max = entry["scale_endpoints"]["max"]

                for item in items_to_recode:
                    df[item] = pd.to_numeric(df[item])
                    if item in df.columns:
                        df[item] = scale_max + scale_min - df[item]
                    else:
                        raise KeyError(f"col {item} not found in {self.dataset} df")

        return df

    def create_scale_means(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates scale means for specified item columns and adds them as new columns to the DataFrame.

        For each configuration entry, this method computes the mean of the specified item columns
        for the current dataset (`self.dataset`). A new column is created with the name defined in the
        configuration (`name` key), storing the calculated scale mean.

        Args:
            df: A pandas DataFrame containing the item columns.

        Returns:
            pd.DataFrame: The DataFrame with new columns added for the scale means.
        """
        cont_pers_entries = self.config_parser(
            self.cfg_preprocessing["person_level"]["personality"], "continuous"
        )
        cont_soc_dem_entries = self.config_parser(
            self.cfg_preprocessing["person_level"]["sociodemographics"], "continuous"
        )
        for entry in cont_pers_entries + cont_soc_dem_entries:
            if self.dataset in entry["item_names"]:
                item_cols = entry["item_names"][self.dataset]
                df[item_cols] = df[item_cols].apply(pd.to_numeric, errors="coerce")

                try:
                    df = df.assign(
                        **{entry["name"]: df[item_cols].mean(axis=1, skipna=True)}
                    )
                except ValueError:
                    df[entry["name"]] = df[item_cols]

        return df

    def store_trait_wb_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """store
        Stores well-being (wb) trait items from the DataFrame for later use.

        This method extracts positive affect (PA) and negative affect (NA) trait items based on the
        configuration, combines them into a list of well-being items (`wb_trait_items`), and saves
        the resulting DataFrame to a file. The input DataFrame is returned unchanged.

        Args:
            df: A pandas DataFrame containing trait-level data.

        Returns:
            pd.DataFrame: The input DataFrame, unchanged.
        """
        try:
            pa_trait_items = self.cfg_preprocessing["person_level"]["criterion"][0][
                "item_names"
            ][self.dataset]
        except KeyError:
            pa_trait_items = []

        try:
            na_trait_items = self.cfg_preprocessing["person_level"]["criterion"][1][
                "item_names"
            ][self.dataset]
        except KeyError:
            na_trait_items = []

        wb_trait_items = pa_trait_items + na_trait_items
        wb_trait_items.append("country")

        if wb_trait_items:
            df_trait_wb_items = df[wb_trait_items]

            if self.cfg_preprocessing["general"]["store_trait_wb_items"]:
                filename = os.path.join(
                    self.cfg_preprocessing["general"]["path_to_preprocessed_data"],
                    f"trait_wb_items_{self.dataset}",
                )
                with open(filename, "wb") as f:
                    pickle.dump(df_trait_wb_items, f)

        return df

    def store_raw_trait_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stores the personality variables where scales are formed for later testing of
        measurement invariance.

        For each trait in the personality configuration:
        - Identifies the survey source (e.g., 'emotions', 'zpid') with more than one item present in the current dataset
        - Extracts item names for those sources
        - Aggregates all such item names for all traits
        - Filters and returns a DataFrame with only these selected item columns

        Args:
            df: A pandas DataFrame containing trait-level data.

        Returns:
            pd.DataFrame: A filtered DataFrame containing only the relevant item variables used for scale formation.
        """
        personality_vars_cfg = self.cfg_preprocessing["person_level"]["personality"]
        relevant_items = ["country"]

        for trait in personality_vars_cfg:
            if self.dataset in trait["item_names"]:
                item_names = trait["item_names"][self.dataset]
                if len(item_names) > 1:
                    relevant_items.extend(item_names)

        if relevant_items:
            raw_trait_df = df[relevant_items].copy()
            filename = os.path.join(
                self.cfg_preprocessing["general"]["path_to_preprocessed_data"],
                f"raw_trait_df_{self.dataset}",
            )
            with open(filename, "wb") as f:
                pickle.dump(raw_trait_df, f)

        return df

    def create_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates items to create six well-being criteria: state and trait-level PA, NA, and WB.

        This method calculates six criteria based on trait and state well-being (WB), positive affect (PA),
        and negative affect (NA). The criteria are derived by averaging across the respective items for PA and NA.
        The WB score is calculated by combining the PA score and an inversely coded NA score, except in specific
        cases like `zpid`, where the WB score is directly assigned from the PA score (i.e., valence).

        **Criteria Created**:
        - `state_wb`, `state_pa`, `state_na`
        - `trait_wb`, `trait_pa`, `trait_na`

        Steps:
        1. Retrieve configuration for state and trait criteria.
        2. For each criterion type (`trait`, `state`):
            - Calculate PA and NA by averaging respective items.
            - Inverse code the NA score using scale endpoints and combine it with PA to compute WB.
            - Handle exceptions (e.g., the `zpid` dataset directly assigns PA to WB).
        3. Log the mean and standard deviation for each calculated criterion.

        Args:
            df: A pandas DataFrame containing well-being item data for individuals.

        Returns:
            pd.DataFrame: The modified DataFrame with additional columns for the six calculated criteria.

        Raises:
            KeyError: If required items for PA or NA are not found in the configuration.
        """
        criteria_types = {"trait": "person_level", "state": "esm_based"}
        for criterion_type, config_lvl in criteria_types.items():
            wb_items = self.config_parser(
                self.cfg_preprocessing[config_lvl]["criterion"], var_type=None
            )
            pa_items = None
            na_items = None

            if self.dataset == "emotions" and criterion_type == "trait":
                continue

            if self.dataset in wb_items[0]["item_names"]:  # pa
                pa_items = wb_items[0]["item_names"][self.dataset]
                df[f"pa_{criterion_type}"] = df[pa_items].mean(axis=1)
                self.logger.log(
                    f"    M {criterion_type}_pa: {np.round(np.mean(df[f'pa_{criterion_type}']), 3)}"
                )
                self.logger.log(
                    f"    SD {criterion_type}_pa: {np.round(np.std(df[f'pa_{criterion_type}']), 3)}"
                )

            if self.dataset in wb_items[1]["item_names"]:  # na
                na_items = wb_items[1]["item_names"][self.dataset]
                df[f"na_{criterion_type}"] = df[na_items].mean(axis=1)
                self.logger.log(
                    f"    M {criterion_type}_na: {np.round(np.mean(df[f'na_{criterion_type}']), 3)}"
                )
                self.logger.log(
                    f"    SD {criterion_type}_na: {np.round(np.std(df[f'na_{criterion_type}']), 3)}"
                )

            # wb
            scale_min = wb_items[1]["scale_endpoints"]["min"]
            scale_max = wb_items[1]["scale_endpoints"]["max"]
            if na_items:
                df[f"{criterion_type}_na_tmp"] = inverse_code(
                    df[na_items], min_scale=scale_min, max_scale=scale_max
                ).mean(axis=1)
                new_columns = pd.DataFrame(
                    {
                        f"wb_{criterion_type}": df[
                            [f"pa_{criterion_type}", f"{criterion_type}_na_tmp"]
                        ].mean(axis=1)
                    }
                )
                df = pd.concat([df, new_columns], axis=1)
                df = df.drop([f"{criterion_type}_na_tmp"], axis=1)
            else:  # zpid
                df[f"wb_{criterion_type}"] = df[f"pa_{criterion_type}"]
                df = df.drop(f"pa_{criterion_type}", axis=1)

            self.logger.log(
                f"    M {criterion_type}_wb: {np.round(np.mean(df[f'wb_{criterion_type}']), 3)}"
            )
            self.logger.log(
                f"    SD {criterion_type}_wb: {np.round(np.std(df[f'wb_{criterion_type}']), 3)}"
            )

        return df

    def sanity_checking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes sanity checks on the DataFrame.

        This method uses the `sanity_checker` component to run predefined sanity checks on the
        preprocessed Dataframe (e.g., checking the number of rows/columns, computing Cronbachs alpha,...).
        To do so, it uses the final DataFrame and the preliminary DataFrame that still contains the individual
        items used for creating the scale means.

        Args:
            df: A pandas DataFrame to be checked.

        Returns:
            pd.DataFrame: The input DataFrame, unchanged.
        """
        self.sanity_checker.run_sanity_checks(
            df=df,
            dataset=self.dataset,
            df_before_final_sel=self.df_before_final_selection,
        )
        return df

    def select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selects and renames the final columns to be included in the output DataFrame.

        This method filters the input DataFrame to include only the specified columns based on the
        configuration in `self.cfg_preprocessing["var_assignments"]`. It also prefixes the selected columns
        with category-specific prefixes (e.g., "pl_" for person-level variables) for easier identification.

        Included categories of features :
        - Person-level variables (`pl`)
        - Self-reported micro context (`srmc`)
        - Sensed micro context (`sens`)
        - Macro context (`macro`)

        Args:
            df: A pandas DataFrame containing columns from different categories.

        Returns:
            pd.DataFrame: A DataFrame with selected columns renamed with their respective prefixes.

        Raises:
            KeyError: If a specified column is not found in the input DataFrame.
        """
        final_df = pd.DataFrame()

        for prefix, columns in self.cfg_preprocessing["var_assignments"].items():
            selected_columns = [col for col in columns if col in df.columns]
            renamed_columns = {col: f"{prefix}_{col}" for col in selected_columns}

            try:
                prefixed_df = df[selected_columns].rename(columns=renamed_columns)
                final_df = pd.concat([final_df, prefixed_df], axis=1)
            except KeyError:
                self.logger.log(
                    f"    WARNING: Some columns of {selected_columns} are not present in {self.dataset} df"
                )

        return final_df

    @staticmethod
    def fill_unique_id_col(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates or updates a "unique_id" column in the DataFrame.

        This method ensures the presence of a "unique_id" column in the DataFrame. For datasets
        other than `cocoms`, it assigns the DataFrame's indices as the values in the "unique_id"
        column. If the column already exists, missing values are filled with the corresponding
        index values.

        This ensures compatibility with `GroupKFold` during cross-validation, where only the
        `cocoms` dataset requires special handling.

        Args:
            df: A pandas DataFrame for which the "unique_id" column is created or updated.

        Returns:
            pd.DataFrame: The DataFrame with the "unique_id" column filled or created.
        """
        if "unique_id" in df.columns:
            df["unique_id"] = df["unique_id"].fillna(df.index.to_series())
        else:
            df["unique_id"] = df.index

        return df

    def process_and_merge_sensing_data(
        self, sensing_dct: dict[str, pd.DataFrame], df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes and merges sensing data with person-level data.

        This method processes multiple sensing datasets to create person-level variables and merges them
        with the person-level data. The steps include merging, column selection, dataset-specific processing,
        applying cut-offs, and creating person-level descriptive statistics.

        **Steps Performed**:
        1. **Merge Sensing DataFrames**: Combines multiple sensing datasets into a single DataFrame.
        2. **Select Relevant Columns**: Filters columns based on the configuration (`df_type = "sensing_based"`).
        3. **Dataset-Specific Processing**: Applies custom preprocessing based on the dataset, including handling NaNs.
        4. **Convert Datetime to Minutes**: Converts datetime columns (i.e., `daily_sunset`, `daily_sunrise`) to minute values.
        5. **Filter Usage Windows**: Filters based on usage periods (`first_usage`, `last_usage`).
        6. **Apply Cut-Offs**: Removes values outside defined thresholds based on configuration.
        7. **Set NaN to Zero for Apps**: Handles NaN values for app data, setting them to zero for unused apps.
        8. **Set NaN to Zero for Calls**: Handles NaN values for call data, setting them to zero for no calls per day.
        9. **Run Sanity Checks**: Verifies data integrity using some sanity checks specifically for the sensing data.
        10. **Create Person-Level Descriptive Stats**: Aggregates sensing variables to create person-level statistics.
        11. **Collapse to Person-Level Data**: Reduces the sensing DataFrame to one row per person.

        Finally, the processed sensing data is merged with the main DataFrame.

        Args:
            sensing_dct: A dictionary of sensing DataFrames, keyed by sensing variable types.
            df: A pandas DataFrame containing participant-level data.

        Returns:
            pd.DataFrame: The participant-level DataFrame enriched with processed sensing data.
        """
        df_sensing = None

        steps = [
            (self.merge_sensing_dfs, {"sensing_dct": sensing_dct}),
            (self.select_columns, {"df": None, "df_type": "sensing_based"}),
            (
                self.dataset_specific_sensing_processing,
                {
                    "df_sensing": None,
                },
            ),
            (
                self.change_datetime_to_minutes,
                {"df": None, "var1": "daily_sunset", "var2": "daily_sunrise"},
            ),
            (
                self.filter_first_last_screen,
                {"df": None, "var1": "first_usage", "var2": "last_usage"},
            ),
            (self.apply_cut_offs, {"df": None}),
            (
                self.set_nan_to_zero,
                {
                    "df": None,
                    "selected_cols_part": self.cfg_preprocessing["general"]["sensing"][
                        "app_substring"
                    ][self.dataset],
                    "reference": "dummy",
                },
            ),
            (
                self.set_nan_to_zero,
                {
                    "df": None,
                    "selected_cols_part": self.cfg_preprocessing["general"]["sensing"][
                        "call_substring"
                    ][self.dataset],
                    "reference": self.cfg_preprocessing["general"]["sensing"][
                        "call_reference"
                    ][self.dataset],
                },
            ),
            (
                self.sanity_checker.sanity_check_sensing_data,
                {"df_sensing": None, "dataset": self.dataset},
            ),
            (
                self.create_person_level_desc_stats,
                {"df": None, "feature_category": "sensing_based"},
            ),
            (self.collapse_df, {"df": None, "df_type": "sensing_based"}),
        ]

        for method, kwargs in steps:
            kwargs = {k: v if v is not None else df_sensing for k, v in kwargs.items()}
            df_sensing = self._log_and_execute(method, indent=6, **kwargs)

        df = self.merge_state_df_sensing(df=df, df_sensing=df_sensing)
        return df

    def dataset_specific_sensing_processing(
        self, df_sensing: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Applies dataset-specific preprocessing to the sensing data.

        This method is a placeholder intended to be overridden in subclasses to handle
        any dataset-specific preprocessing requirements for sensing data. The base
        implementation returns the input DataFrame unchanged.

        Args:
            df_sensing: A pandas DataFrame containing sensing data.

        Returns:
            pd.DataFrame: The sensing DataFrame after any dataset-specific processing
            (unchanged in the base implementation).
        """
        return df_sensing

    def merge_sensing_dfs(self, sensing_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges two sensing DataFrames (mobile phone and GPS + weather features).

        This method concatenates the two sensing DataFrames along columns using a common user ID
        and date column. The method ensures proper formatting of IDs and dates to facilitate
        accurate merging.

        Args:
            sensing_dct: A dictionary containing sensing DataFrames keyed as:
                - "phone_sensing": DataFrame with mobile phone sensing features.
                - "gps_weather": DataFrame with GPS and weather features.

        Returns:
            pd.DataFrame: A merged DataFrame containing both mobile phone and GPS + weather features.
        """
        date_col_phone = self.cfg_preprocessing["sensing_based"]["phone"][1][
            "item_names"
        ][self.dataset]
        date_col_gps_weather = self.cfg_preprocessing["sensing_based"]["gps_weather"][
            1
        ]["item_names"][self.dataset]

        phone_df = sensing_dct["phone_sensing"]
        gps_weather_df = sensing_dct["gps_weather"]
        phone_df["user_id"] = pd.to_numeric(phone_df.user_id)
        gps_weather_df["user_id"] = pd.to_numeric(gps_weather_df.user_id)

        phone_df[date_col_phone] = pd.to_datetime(
            phone_df[date_col_phone], format="mixed", errors="coerce"
        )
        gps_weather_df[date_col_gps_weather] = pd.to_datetime(
            gps_weather_df[date_col_gps_weather], format="mixed", errors="coerce"
        )

        merged_df = pd.merge(
            phone_df,
            gps_weather_df,
            left_on=[self.raw_sensing_id_col, date_col_phone],
            right_on=[self.raw_sensing_id_col, date_col_gps_weather],
            how="outer",
        )

        return merged_df

    def change_datetime_to_minutes(
        self, df: pd.DataFrame, **kwargs: dict[str, str]
    ) -> pd.DataFrame:
        """
        Converts datetime columns to total minutes.

        Note: Args makes more sense than kwargs here, but this is necessary to be integrated in the
        argparsing structure of 'process_and_merge_sensing_data'

        Args:
            df: A pandas DataFrame containing the datetime columns to be processed.
            **kwargs: Keyword arguments representing variable names, where keys are unused,
                      and values are column-related variable names from the configuration.

        Returns:
            pd.DataFrame: The modified DataFrame with datetime columns converted to minutes.
        """

        for var_name in kwargs.values():
            col_name = self.config_parser(
                self.cfg_preprocessing["sensing_based"]["gps_weather"],
                "continuous",
                var_name,
            )[0]["item_names"][self.dataset]
            df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
            df[col_name] = df[col_name].dt.hour * 60 + df[col_name].dt.minute

        return df

    def filter_first_last_screen(self, df: pd.DataFrame, **kwargs: str) -> pd.DataFrame:
        """
        Filters specific columns to exclude certain values.

        This method processes the columns representing first and last phone usage, setting values between 0 and 10, or between
        1430 and 1440, to `np.nan` as these values may be prone to errors.
        The column names are retrieved based on `kwargs` and the configuration.

        Note: Args makes more sense than kwargs here, but this is necessary to be integrated in the
        argparsing structure of 'process_and_merge_sensing_data'

        Args:
            df: A pandas DataFrame containing the columns to be filtered.
            **kwargs: Keyword arguments representing variable names, where values correspond
                      to the column keys in the configuration.

        Returns:
            pd.DataFrame: The modified DataFrame with filtered columns.
        """
        for var_name in kwargs.values():
            col_name = self.config_parser(
                self.cfg_preprocessing["sensing_based"]["phone"], "continuous", var_name
            )[0]["item_names"][self.dataset]
            df[col_name] = df[col_name].where(
                ~(
                    (df[col_name] >= 0) & (df[col_name] <= 10)
                    | (df[col_name] >= 1430) & (df[col_name] <= 1440)
                ),
                np.nan,
            )
        return df

    def apply_cut_offs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies threshold-based cut-offs to sensing variables.

        This method filters sensing variables by capping their values at a predefined threshold,
        specified in the configuration. For example, if the threshold for a variable is 1440,
        any values above 1440 are set to 1440.

        Args:
            df: A pandas DataFrame containing sensing variables to be processed.

        Returns:
            pd.DataFrame: The modified DataFrame with capped values for variables with defined thresholds.
        """
        vars_phone_sensing = self.config_parser(
            self.cfg_preprocessing["sensing_based"]["phone"], "continuous"
        )
        vars_gps_weather = self.config_parser(
            self.cfg_preprocessing["sensing_based"]["gps_weather"], "continuous"
        )
        total_vars = vars_phone_sensing + vars_gps_weather

        for sens_var in total_vars:
            if "cut_off" in sens_var:
                col = sens_var["item_names"][self.dataset]
                df[col] = df[col].clip(upper=sens_var["cut_off"])
                self.logger.log(f"    Apply Cut-Off for var {sens_var['name']}")

        return df

    def merge_state_df_sensing(
        self, df: pd.DataFrame, df_sensing: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges sensing data with state-level data.

        This method combines the participant-level DataFrame (`df`) with the sensing DataFrame (`df_sensing`)
        using an outer join. The merging logic ensures flexibility for analyses with and without sensing data:
        - Samples with only questionnaire data are retained.
        - Samples with only ESM and sensing data are also included, if applicable.
        - Missing trait data for individuals with ESM or sensing data is accommodated.

        Args:
            df: A pandas DataFrame containing participant-level state or trait data.
            df_sensing: A pandas DataFrame containing participant-level sensing data.

        Returns:
            pd.DataFrame: The merged DataFrame, combining state/trait data with sensing data.
        """
        df = pd.merge(
            left=df,
            right=df_sensing,
            left_on=[self.raw_esm_id_col],
            right_on=[self.raw_sensing_id_col],
            how="left",
        )
        return df

    def set_nan_to_zero(
        self, df, selected_cols_part: str, reference: str = None
    ) -> pd.DataFrame:
        """
        Fills NaN values with zero in selected columns based on specified conditions.

        This method identifies columns containing a specific substring (`selected_cols_part`)
        and processes their rows as follows:
        - If all values in the selected columns (and optionally a reference column) are NaN,
          NaNs are left unchanged.
        - If at least one value in the selected columns (or the reference column, if specified)
          is not NaN, NaNs in the selected columns are replaced with 0.

        Args:
            df: A pandas DataFrame to process.
            selected_cols_part: A substring to identify target columns for processing.
            reference: An optional column used to determine whether NaNs in the selected
                       columns should be replaced with 0. The reference column itself remains unchanged.

        Returns:
            pd.DataFrame: The processed DataFrame with updated NaN values in the selected columns.
        """
        selected_cols = [col for col in df.columns if selected_cols_part in col]
        self.logger.log(f"    Fill NaN with zeros in: {selected_cols}")

        def process_row(row: pd.Series) -> pd.Series:
            """
            Processes a single row to update NaN values in the selected columns.

            If all relevant columns (selected columns and the optional reference column)
            are NaN, the NaNs in the selected columns remain unchanged. Otherwise, NaNs
            in the selected columns are replaced with 0.

            Args:
                row: A pandas Series representing a single row from the DataFrame.

            Returns:
                pd.Series: The processed row with updated NaN values in the selected columns.
            """
            if reference in df.columns:
                if row[selected_cols + [reference]].isna().all():
                    return row[selected_cols]
                else:
                    return row[selected_cols].fillna(0).infer_objects(copy=False)

            else:
                if row[selected_cols].isna().all():
                    return row[selected_cols]
                else:
                    return row[selected_cols].fillna(0).infer_objects(copy=False)

        df[selected_cols] = df.apply(process_row, axis=1)

        return df
