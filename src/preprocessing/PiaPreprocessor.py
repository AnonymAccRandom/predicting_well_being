import re

import numpy as np
import pandas as pd

from src.preprocessing.BasePreprocessor import BasePreprocessor
from src.utils.utilfuncs import NestedDict


class PiaPreprocessor(BasePreprocessor):
    """
    Preprocessor for the "pia" dataset, inheriting from BasePreprocessor.

    This class implements preprocessing logic specific to the "pia" dataset.
    It inherits all the attributes and methods of BasePreprocessor, including:
    - Configuration file ('cfg_preprocessing').
    - Logging and timing utilities (`logger`, `timer`).
    - Data loading, processing, and sanity checking methods.

    Additional Attributes (for other Attributes, see BasePreprocessor):
        dataset (str): Specifies the current dataset as "pia".
    """

    def __init__(self, cfg_preprocessing: NestedDict) -> None:
        """
        Initializes the PiaPreprocessor with dataset-specific configurations.

        Args:
            cfg_preprocessing: Yaml config specifying details on preprocessing (e.g., scales, items).
        """
        super().__init__(cfg_preprocessing=cfg_preprocessing)
        self.dataset = "pia"

    def merge_traits(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Retrieves the DataFrame for trait-level data from the input dictionary.

        Args:
            df_dct: A dictionary containing multiple DataFrames,
                    with keys indicating their types.

        Returns:
            pd.DataFrame: The DataFrame containing trait-level data.
        """
        return df_dct["data_traits"]

    def dataset_specific_trait_processing(
        self, df_traits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes trait data specific to PIA by creating and populating new columns:
         - "country": Assigned a constant value of "germany".

        Args:
            df_traits: The input DataFrame containing trait-level data.

        Returns:
            pd.DataFrame: The modified DataFrame with the added 'country' column.
        """
        df_traits["country"] = "germany"
        return df_traits

    def dataset_specific_state_processing(
        self, df_states: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes state data specific to PIA by creating and populating new columns:
         - "country": Assigned a constant value of "germany".

        Args:
            df_traits: The input DataFrame containing trait-level data.

        Returns:
            pd.DataFrame: The modified DataFrame with the added 'country' column.
        """
        df_states["country"] = "germany"
        return df_states



    def merge_states(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Retrieves the DataFrame for state-level data from the input dictionary.

        Args:
            df_dct: A dictionary containing multiple DataFrames, with keys indicating their types.

        Returns:
            pd.DataFrame: The DataFrame containing state-level data.
        """
        return df_dct["data_esm"]

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate columns caused by suffixes and consolidates missing values.

        This function processes columns with suffixes (e.g., '_t2', '_t3', '_t4'). Missing values in the
        base column are filled with values from higher-suffix columns in descending order of suffix priority.
        If a base column does not exist, it is created using the first available suffix column.

        Furthermore, it removes the 'r' from column names that match a regex pattern of a number followed by 'r'
        (e.g., "10r" for 10th item recoded).

        Args:
            df_traits: A pandas DataFrame containing trait-level data with possible duplicates.

        Returns:
            pd.DataFrame: The updated DataFrame with consolidated column values and no duplicates.

        Raises:
            AssertionError: If duplicate column names remain after renaming.
        """
        trait_suffixes = self.cfg_preprocessing["general"]["pl_suffixes"][self.dataset]
        suffix_pattern = re.compile(r"_t\d$")

        for col in df_traits.columns:
            base_col = re.sub(suffix_pattern, "", col)

            if base_col not in df_traits.columns:
                df_traits[base_col] = np.nan

            for suffix in trait_suffixes:  # Start with '_t2', then '_t3', and '_t4'
                suffix_col = f"{base_col}{suffix}"

                if suffix_col in df_traits.columns:
                    df_traits[base_col].fillna(df_traits[suffix_col], inplace=True)
                    df_traits = df_traits.drop(columns=suffix_col)

        assert len(df_traits.columns) == len(
            set(df_traits.columns)
        ), "Duplicate column names found after renaming!"

        return df_traits
