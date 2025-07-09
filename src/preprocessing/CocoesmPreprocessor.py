import re
from copy import deepcopy

import numpy as np
import pandas as pd

from src.preprocessing.BasePreprocessor import BasePreprocessor
from src.utils.utilfuncs import NestedDict


class CocoesmPreprocessor(BasePreprocessor):
    """
    Preprocessor for the "cocoesm" dataset, inheriting from BasePreprocessor.

    This class implements preprocessing logic specific to the "cocoesm" dataset.
    It inherits all the attributes and methods of BasePreprocessor, including:
    - Configuration file ('cfg_preprocessing').
    - Logging and timing utilities (`logger`, `timer`).
    - Data loading, processing, and sanity checking methods.

    Additional Attributes (for other Attributes, see BasePreprocessor):
        dataset (str): Specifies the current dataset as "cocoesm".
        relationship (Optional[pd.DataFrame]): Reserved for storing relationship-specific data, assigned during processing.
    """

    def __init__(self, cfg_preprocessing: NestedDict) -> None:
        """
        Initializes the CoCoESM Preprocessor with dataset-specific configurations.

        Args:
            cfg_preprocessing: Yaml config specifying details on preprocessing (e.g., scales, items).
        """
        super().__init__(cfg_preprocessing=cfg_preprocessing)
        self.dataset = "cocoesm"
        self.relationship = None  # will be assigned stored for later use

    def merge_traits(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges the trait data from the provided dictionary of DataFrames.

        This method extracts and returns the "data_traits" DataFrame from the input dictionary. No special
        preprocessing necessary for 'cocoesm'.

        Args:
            df_dct: A dictionary where keys are dataset names and values are DataFrames.

        Returns:
            pd.DataFrame: The DataFrame containing trait data.
        """
        return df_dct["data_traits"]

    def merge_states(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges ESM and daily ESM data into a single DataFrame.

        The method processes the `data_esm` and `data_esm_daily` DataFrames by:
        - Parsing the `created_individual` column to extract the date.
        - Merging the two DataFrames on the `participant` and `date` columns using a left join.
        - Adding suffixes to distinguish overlapping columns from both DataFrames.

        Args:
            df_dct: A dictionary containing `data_esm` and `data_esm_daily` DataFrames.

        Returns:
            pd.DataFrame: The merged DataFrame with ESM and daily ESM data combined.
        """
        data_esm = df_dct["data_esm"]
        data_esm_daily = df_dct["data_esm_daily"]
        data_esm["date"] = pd.to_datetime(data_esm["created_individual"]).dt.date
        data_esm_daily["date"] = pd.to_datetime(
            data_esm_daily["created_individual"]
        ).dt.date

        merged_df = pd.merge(
            data_esm,
            data_esm_daily,
            how="left",
            on=["participant", "date"],
            suffixes=("_esm", "_daily"),
        )

        return merged_df

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans trait DataFrame column names by:
        - Removing a specified suffix (e.g., "_t1") from column names if present.
        - Removing the 'r' from column names that match a regex pattern of a number followed by 'r'
            (e.g., "10r" for 10th item recoded).

        The suffix pattern is defined in the configuration (`cfg_preprocessing`).

        Args:
            df_traits: The DataFrame containing trait data with column names to be updated.

        Returns:
            pd.DataFrame: A DataFrame with cleaned column names.
        """
        trait_suffix = self.cfg_preprocessing["general"]["pl_suffixes"]["cocoesm"]
        regex_pattern: str = r"(\d)r$"
        updated_columns = []

        for col in df_traits.columns:
            if col.endswith(trait_suffix):
                col = col[: -len(trait_suffix)]
            col = re.sub(regex_pattern, r"\1", col)
            updated_columns.append(col)
        df_traits.columns = updated_columns

        return df_traits

    def dataset_specific_trait_processing(
        self, df_traits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Applies dataset-specific processing to the trait DataFrame.

        For the "CoCo ESM" dataset, this includes:
        - Setting the `relationship_household` column to '0' for rows where `quantity_household` equals 1.

        Args:
            df_traits: The DataFrame containing trait data.

        Returns:
            pd.DataFrame: The modified DataFrame after applying dataset-specific processing.
        """
        df_traits.loc[
            df_traits["quantity_household"] == 1, "relationship_household"
        ] = "0"
        return df_traits

    def dataset_specific_state_processing(
        self, df_states: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Applies dataset-specific processing to the state DataFrame.

        For the "CoCo ESM" dataset, this includes:
        - Creating a relationship column using the `create_relationship` method.

        Args:
            df_states: The DataFrame containing state data.

        Returns:
            pd.DataFrame: The modified DataFrame after applying dataset-specific processing.
        """
        df_states = self._create_relationship(df_states=df_states)
        df_states = df_states.rename(columns={"country_esm": "country"})

        return df_states

    def _create_relationship(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Infers relationship status from ESM survey data by analyzing interactions with a partner.

        If any row for a person has a specific value (e.g., `4`) in the partner interaction column (as defined in the
        configuration), the person is considered to be in a relationship. The relationship status is stored in a
        new column, `relationship`.

        Args:
            df_states: The DataFrame containing ESM survey data.

        Returns:
            pd.DataFrame: The modified DataFrame with a new `relationship` column.
        """
        relationship_cfg = self.config_parser(
            self.cfg_preprocessing["esm_based"]["self_reported_micro_context"],
            "binary",
            "relationship",
        )[0]
        ia_partner_col = relationship_cfg["item_names"]["cocoesm"]
        ia_partner_val = relationship_cfg["special_mappings"]["cocoesm"]

        if ia_partner_col in df_states.columns:
            df_states["interaction"] = df_states[ia_partner_col].apply(
                lambda x: self._map_comma_separated(x, {ia_partner_val: 1})
            )
            partner_interaction = df_states.groupby(self.raw_esm_id_col)[
                "interaction"
            ].transform("max")
            df_states["relationship"] = np.where(partner_interaction == 1, 1, 0)
            df_states.drop(columns=["interaction"], inplace=True)

        else:
            raise KeyError(f"Column {ia_partner_col} not in {self.dataset}")

        self.relationship = deepcopy(
            df_states[["relationship", self.raw_esm_id_col]].drop_duplicates(
                keep="first"
            )
        )
        return df_states

    def dataset_specific_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies post-processing steps specific to the dataset.

        For "CoCo ESM," this includes:
        - Merging the `relationship` information into the DataFrame.
        - Filling missing country-level data using the `fill_country_nans` method.

        Args:
            df: The DataFrame containing processed data.

        Returns:
            pd.DataFrame: The modified DataFrame after applying post-processing.
        """
        df = df.merge(self.relationship, on=self.raw_esm_id_col, how="left")
        return df

