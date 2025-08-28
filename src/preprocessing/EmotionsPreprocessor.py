import re
from copy import deepcopy

import numpy as np
import pandas as pd

from src.preprocessing.BasePreprocessor import BasePreprocessor
from src.utils.utilfuncs import NestedDict


class EmotionsPreprocessor(BasePreprocessor):
    """
    Preprocessor for the "emotions" dataset, inheriting from BasePreprocessor.

    This class implements preprocessing logic specific to the "emotions" dataset.
    It inherits all the attributes and methods of BasePreprocessor, including:
    - Configuration file ('cfg_preprocessing').
    - Logging and timing utilities (`logger`, `timer`).
    - Data loading, processing, and sanity checking methods.

    Additional Attributes (for other Attributes, see BasePreprocessor):
        dataset (str): Specifies the current dataset as "emotions".
        close_interactions (Any): Stores data related to close interactions, assigned during preprocessing.
    """

    def __init__(self, cfg_preprocessing: NestedDict) -> None:
        """
        Initializes the EmotionsPreprocessor with dataset-specific configurations.

        Args:
            cfg_preprocessing: Yaml config specifying details on preprocessing (e.g., scales, items).
        """
        super().__init__(cfg_preprocessing=cfg_preprocessing)
        self.dataset = "emotions"
        self.close_interactions = None

    def merge_traits(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges DataFrames from the input dictionary that have 'traits' in their key.

        This function processes the 'data_traits_esm' DataFrame (which contains trait and ESM measures)
        to remove duplicates and retain only the first occurrence of each unique ID based on the 'raw_trait_id_col',
        so that we keep only one variable per person.

        Args:
            df_dct: A dictionary where the keys are strings and the values are pandas DataFrames.
                    The DataFrame identified as 'data_traits_esm' is processed for merging.

        Returns:
            pd.DataFrame: A single DataFrame containing the 'traits' data with duplicates removed.
        """
        trait_esm_df = df_dct["data_traits_esm"]
        df_traits = trait_esm_df.drop_duplicates(
            subset=self.raw_trait_id_col, keep="first"
        ).reset_index(drop=True)

        return df_traits

    def clean_trait_col_duplicates(
        self, df_traits: pd.DataFrame
    ) -> pd.DataFrame:  # TODO: Is this in PreReg / Paper? Check
        """
        Cleans up column names in a trait DataFrame by handling duplicates and suffixes.

        This method performs the following:
        - Removes specified suffixes (e.g., '_t2', '_t3', '_t4') from column names.
        - Fills missing values in the base column (e.g., 'narqs_1') using values from corresponding suffix columns
          (e.g., 'narqs_1_t2', 'narqs_1_t3').
        - If a base column does not exist, creates it by filling it with values from the first available suffix column.
        - Drops suffix columns after transferring their data to the base column.
        - Ensures no duplicate column names remain in the DataFrame.

        Args:
            df_traits: A pandas DataFrame containing trait data with column names potentially ending
                       with suffixes such as '_t2', '_t3', or '_t4'.

        Returns:
            pd.DataFrame: The cleaned DataFrame with updated column names and filled missing values.

        Raises:
            AssertionError: If duplicate column names are found after renaming and processing.
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
                    if df_traits[base_col].isna().any:
                        df_traits[base_col].fillna(df_traits[suffix_col], inplace=True)
                        df_traits = df_traits.drop(columns=suffix_col)

        assert len(df_traits.columns) == len(
            set(df_traits.columns)
        ), "Duplicate column names found after renaming!"

        return df_traits

    def dataset_specific_trait_processing(
        self, df_traits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes trait data specific to Emotions by creating and populating new columns:
         - "country": Assigned a constant value of "germany".

        Args:
            df_traits: The input DataFrame containing trait-level data.

        Returns:
            pd.DataFrame: The modified DataFrame with the added 'country' column.
        """
        df_traits["country"] = "germany"
        return df_traits

    def merge_states(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges state-level DataFrames from the input dictionary and filters columns to retain
        only those that vary across rows for the same participant.

        Processing steps:
        - Selects the DataFrame associated with state-level data from the input dictionary.
        - Identifies columns where values differ across rows for the same participant ('id_for_merging').
        - Retains only these varying columns, along with the participant ID and 'wave' columns.

        Args:
            df_dct: A dictionary where keys are strings and values are pandas DataFrames.
                    The state-level data is identified using specific key naming conventions.

        Returns:
            pd.DataFrame: A filtered DataFrame containing state-level data with columns that vary
                          across rows for the same participant.
        """
        state_df = df_dct["data_traits_esm"]
        grouped = state_df.groupby(self.raw_esm_id_col)

        varying_columns = [
            col for col in state_df.columns if grouped[col].nunique().gt(1).any()
        ]
        state_df_filtered = state_df[varying_columns + [self.raw_esm_id_col] + ["wave"]]

        return state_df_filtered

    def dataset_specific_state_processing(
        self, df_states: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Applies dataset-specific processing steps to state-level data.

        Processing steps:
        1. Creates a column for close interactions using `create_close_interactions`.
        2. Merges interaction and occupational states using `merge_int_occup_states`.

        Args:
            df_states: The pandas DataFrame containing state-level data.

        Returns:
            pd.DataFrame: The processed DataFrame with updated columns specific to the dataset.
        """
        df_states = self.create_close_interactions(df_states=df_states)
        df_states = self.merge_int_occup_states(df_states=df_states)
        df_states["country"] = "germany"

        return df_states

    def create_close_interactions(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies and calculates the percentage of close interactions in the Emotions dataset.

        In Emotions, each prompt refers to up to 5 interaction partners
        (interaction_partner1 - interaction_partner5). An interaction is classified as a
        close tie if all interaction partners are close ties, and as a weak tie if all
        interaction partners are weak ties. Interactions with mixed or unclassifiable ties
        are set to NaN.

        Processing steps:
        1. Replace interaction partner categories with mapped categorical values.
        2. Identify prompts where all interaction partners are close ties or weak ties.
        3. Create the `close_interactions_raw` column to classify each prompt.
        4. Calculate the percentage of close interactions per person.
        5. Store the results in `self.close_interactions` for later use.

        Args:
            df_states: The DataFrame containing state-level interaction data.

        Returns:
            pd.DataFrame: The updated DataFrame with new columns for close interactions.

        Raises:
            KeyError: If any required interaction partner columns are missing in the DataFrame.
        """
        close_interaction_cfg = [
            entry
            for entry in self.cfg_preprocessing["esm_based"][
                "self_reported_micro_context"
            ]
            if entry["name"] == "close_interactions"
        ][0]
        int_partner_cols = close_interaction_cfg["special_mappings"]["emotions"][
            "columns"
        ]
        cat_mapping = close_interaction_cfg["special_mappings"]["emotions"]["mapping"]

        for col in int_partner_cols:
            df_states[col] = df_states[col].replace(cat_mapping)

        # Close interaction partner is 1, 0 is weak
        all_close_mask = df_states[int_partner_cols].isin([1, np.nan]).all(axis=1)
        all_weak_mask = df_states[int_partner_cols].isin([0, np.nan]).all(axis=1)
        df_states["close_interactions_raw"] = np.where(
            all_close_mask, 1, np.where(all_weak_mask, 0, np.nan)
        )

        interaction_stats = df_states.groupby(self.raw_esm_id_col)[
            "close_interactions_raw"
        ].apply(lambda x: x.sum() / x.count() if x.count() > 0 else np.nan)

        df_states["close_interactions"] = df_states[self.raw_esm_id_col].map(
            interaction_stats
        )
        self.close_interactions = deepcopy(
            df_states[["close_interactions", self.raw_esm_id_col]].drop_duplicates(
                keep="first"
            )
        )

        return df_states

    def merge_int_occup_states(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Merges columns for state-level positive and negative affect based on interaction context.

        In the Emotions dataset, separate columns for positive and negative affect are provided
        depending on whether an interaction occurred (`int_`) or not (`occup_`). This method consolidates
        these columns to avoid biases in summaries by filling missing values in one column
        with values from the other.

        Args:
            df_states: The DataFrame containing state-level affect data.

        Returns:
            pd.DataFrame: The updated DataFrame with consolidated affect columns.
        """
        crit_state_cfg = self.cfg_preprocessing["esm_based"]["criterion"]
        pa_items = crit_state_cfg[0]["item_names"][self.dataset]
        na_items = crit_state_cfg[1]["item_names"][self.dataset]

        for item in pa_items + na_items:
            occup_col = f'occup_{item.split("_")[1]}'
            int_col = f'int_{item.split("_")[1]}'

            if occup_col in df_states.columns and int_col in df_states.columns:
                df_states[occup_col] = df_states[occup_col].combine_first(
                    df_states[int_col]
                )
                df_states[int_col] = df_states[occup_col]

        return df_states

    def dataset_specific_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies dataset-specific post-processing for the Emotions dataset.

        This method merges additional features (i.e., close interactions) into the main DataFrame.

        Args:
            df: The main DataFrame after preprocessing.

        Returns:
            pd.DataFrame: The updated DataFrame with merged features.
        """
        df = df.merge(self.close_interactions, on=self.raw_esm_id_col, how="left")
        return df
