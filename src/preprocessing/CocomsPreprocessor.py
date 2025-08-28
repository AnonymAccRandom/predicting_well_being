import re
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd

from src.preprocessing.BasePreprocessor import BasePreprocessor
from src.utils.utilfuncs import NestedDict


class CocomsPreprocessor(BasePreprocessor):
    """
    Preprocessor for the "cocoms" dataset, inheriting from BasePreprocessor.

    This class implements preprocessing logic specific to the "cocoms" dataset.
    It inherits all the attributes and methods of BasePreprocessor, including:
    - Configuration file ('cfg_preprocessing').
    - Logging and timing utilities (`logger`, `timer`).
    - Data loading, processing, and sanity checking methods.

    Additional Attributes (for other Attributes, see BasePreprocessor):
        dataset (str): Specifies the current dataset as "cocoms".
        relationship (Optional[pd.DataFrame]): Stores relationship-specific data for later use.
        work_conversations (Optional[pd.DataFrame]): Stores data related to work conversations.
        personal_conversations (Optional[pd.DataFrame]): Stores data related to personal conversations.
        societal_conversations (Optional[pd.DataFrame]): Stores data related to societal conversations.
        close_interactions: (Optional[pd.DataFrame]): Stores close interaction data.
    """

    def __init__(self, cfg_preprocessing: NestedDict) -> None:
        """
        Initializes the CocomsPreprocessor with dataset-specific configurations.

        Args:
            cfg_preprocessing: Yaml config specifying details on preprocessing (e.g., scales, items).
        """
        super().__init__(cfg_preprocessing=cfg_preprocessing)
        self.dataset = "cocoms"

        self.relationship = None  # will be assigned and stored for later use

        self.work_conversations = None
        self.personal_conversations = None
        self.societal_conversations = None

        self.close_interactions = None

    def merge_traits(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges all trait-related DataFrames in the given dictionary into a single DataFrame.

        This method first applies a preprocessing step specific to CoCoMS using the
        `cocoms_processing_before_merging` method, then concatenates all DataFrames containing
        "traits" in their keys.

        Args:
            df_dct: A dictionary where keys represent data categories
                and values are the corresponding DataFrames.

        Returns:
            pd.DataFrame: A concatenated DataFrame containing all trait data.
        """
        df_dct_proc = self.cocoms_processing_before_merging(deepcopy(df_dct))

        traits_dfs = [df for key, df in df_dct_proc.items() if "traits" in key]
        concatenated_traits = pd.concat(traits_dfs, axis=0).reset_index(drop=True)

        return concatenated_traits

    @staticmethod
    def cocoms_processing_before_merging(
        df_dct: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Applies dataset-specific preprocessing to the CoCoMS trait DataFrames before merging.

        This method performs the following steps:
        - Renames `hex60` columns in wave 3 to match the naming convention in waves 1 and 2 (`hex_60`).
        - Standardizes and aligns the `professional_status` column across waves.
        - Ensures the `professional_status_student_t1` column has consistent binary values across waves.

        Args:
            df_dct: A dictionary of DataFrames where keys represent data categories
                and values are the corresponding DataFrames for different waves (e.g., `data_traits_w1`).

        Returns:
            dict[str, pd.DataFrame]: A dictionary with the processed DataFrames for waves 1, 2, and 3.
        """
        df_w1 = df_dct["data_traits_w1"]
        df_w2 = df_dct["data_traits_w2"]
        df_w3 = df_dct["data_traits_w3"]

        # Rename hexaco cols (hex_60 in w1,w2; hex60 in w3)
        df_w3.columns = [re.sub(r"^hex60", "hex_60", col) for col in df_w3.columns]

        df_w1["professional_status_t1"] = df_w1["professional_status_student_t1"].apply(
            lambda x: 5 if x in [2, 3, 4] else 0
        )
        df_w2["professional_status_t1"] = df_w2["professional_status_student_t1"].apply(
            lambda x: 5 if x in [2, 3, 4] else 0
        )
        df_w3["professional_status_t1"] = df_w3["professional_status_booster_t1"]

        df_w1["professional_status_student_t1"] = 1
        df_w2["professional_status_student_t1"] = 1
        df_w3["professional_status_student_t1"] = pd.notna(
            df_w3["professional_status_student_t1"]
        ).astype(int)

        df_dct["data_esm_w1"] = df_w1
        df_dct["data_esm_w2"] = df_w2
        df_dct["data_esm_w3"] = df_w3

        return df_dct

    def merge_states(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenates state-level ESM DataFrames from a given dictionary into a single DataFrame.

        This method identifies all DataFrames related to ESM (Experience Sampling Method) data
        based on their keys containing the substring 'esm' and combines them along the rows. The
        resulting DataFrame is reset to have a continuous index.

        Args:
            df_dct: A dictionary containing state-level DataFrames, where
                keys indicate the type of data (e.g., `data_esm_w1`, `data_esm_w2`).

        Returns:
            pd.DataFrame: A single concatenated DataFrame containing all ESM-related data.
        """
        esm_dfs = [df for key, df in df_dct.items() if "esm" in key]
        concatenated_esm = pd.concat(esm_dfs, axis=0).reset_index(drop=True)

        return concatenated_esm

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans up trait column names and resolves conflicts between columns with suffixes.

        This method:
        - Strips specified suffixes (e.g., "_t1", "_t2") from column names.
        - For conflicting columns (e.g., "_t1" and "_t2" of the same base name), fills NaN values
          in the "_t1" column with values from the corresponding "_t2" column, and then drops "_t2".

        Args:
            df_traits: The DataFrame containing trait data with suffixes in column names.

        Returns:
            pd.DataFrame: A DataFrame with updated column names and resolved conflicts between suffix columns.
        """
        trait_suffixes = self.cfg_preprocessing["general"]["pl_suffixes"]["cocoms"]
        updated_columns = []
        columns_to_fill = {}

        for col in df_traits.columns:
            original_col = col

            for suffix in trait_suffixes:
                if col.endswith(suffix):
                    base_col_name = col[: -len(suffix)]

                    if suffix == "_t2" and base_col_name in updated_columns:
                        columns_to_fill[base_col_name] = original_col
                        break
                    col = base_col_name

            updated_columns.append(col)

        for base_col, t2_col in columns_to_fill.items():
            t1_col = f"{base_col}_t1"
            if t1_col in df_traits.columns:
                if df_traits[t1_col].isna().any():
                    df_traits[t1_col].fillna(df_traits[t2_col], inplace=True)

        df_traits.columns = updated_columns
        assert len(df_traits.columns) == len(
            set(df_traits.columns)
        ), "Duplicate column names found after renaming!"

        return df_traits

    def dataset_specific_trait_processing(
        self, df_traits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes trait-level data for the CoCoMS dataset.

        In this dataset-specific processing step:
        - Assigns numerical values to elected political parties using a mapping.
        - Scales the assigned party numbers to a consistent range (1–11).
        - Sets the "country" column to "germany".

        Args:
            df_traits: The DataFrame containing trait-level data.

        Returns:
            pd.DataFrame: The updated DataFrame with processed trait-level data.
        """
        party_number_map = [
            entry["party_num_mapping"]
            for entry in self.cfg_preprocessing["person_level"]["personality"]
            if "party_num_mapping" in entry.keys()
        ][0]["cocoms"]

        df_traits["vote_general"] = (
            df_traits["vote_general"].map(party_number_map).fillna(np.nan)
        )
        df_traits["country"] = "germany"

        return df_traits

    def dataset_specific_state_processing(
        self, df_states: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes state-level data specific to the CoCoMS dataset.

        This method applies the following dataset-specific transformations:
        - Creates columns for close interactions from ESM data.
        - Generates conversation topic columns based on ESM data.
        - Infers relationship status based on ESM data.
        - Adjusts the "number_interactions" column for consistency with the other datasets.

        Args:
            df_states: The DataFrame containing state-level data.

        Returns:
            pd.DataFrame: The updated DataFrame after dataset-specific state processing.
        """
        df_states = self.create_close_interactions(df_states=df_states)
        df_states = self.create_conversation_topic_columns(df_states=df_states)
        df_states = self.create_relationship(df_states=df_states)
        df_states = self.adjust_number_interactions_col(df_states=df_states)

        df_states["country"] = "germany"

        return df_states

    def create_close_interactions(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Processes interaction data to derive close interaction metrics for CoCoMS study waves.

        This method creates two columns:
        - "close_interactions_raw": Binary representation where:
          - 1 = close ties,
          - 0 = weak ties,
          - NaN = conflicts or unassessed interactions.
        - "close_interactions": The percentage of close ties relative to total interactions per person.

        Processing by wave:
        - **Wave 1**: Binary columns map unambiguously to close or weak ties.
        - **Wave 2**: No interaction types are assessed; "close_interactions" is NaN.
        - **Wave 3**: Categorical values map to binary in "close_interactions_raw", where -1 (no interaction) is set to NaN.

        Final output includes the ratio of close ties to total ties for each participant.

        Args:
            df_states: The DataFrame containing interaction data for all study waves.

        Returns:
            pd.DataFrame: The updated DataFrame with "close_interactions_raw" and "close_interactions" columns.
        """
        close_interaction_cfg = self.config_parser(
            self.cfg_preprocessing["esm_based"]["self_reported_micro_context"],
            "percentage",
            "close_interactions",
        )[0]
        wave1_weak_ties_col_lst = close_interaction_cfg["special_mappings"]["cocoms"][
            "wave1"
        ]["weak_ties"]
        wave1_close_ties_col_lst = close_interaction_cfg["special_mappings"]["cocoms"][
            "wave1"
        ]["close_ties"]
        wave3_mapping = close_interaction_cfg["special_mappings"]["cocoms"]["wave3"]

        # Wave 1
        close_tie_mask = (
            df_states.loc[df_states["studyWave"] == 1, wave1_close_ties_col_lst].max(
                axis=1
            )
            == 1
        )
        weak_tie_mask = (
            df_states.loc[df_states["studyWave"] == 1, wave1_weak_ties_col_lst].max(
                axis=1
            )
            == 1
        )
        df_states.loc[df_states["studyWave"] == 1, "close_interactions_raw"] = np.where(
            close_tie_mask & ~weak_tie_mask,
            1,
            np.where(weak_tie_mask & ~close_tie_mask, 0, np.nan),
        )

        # Wave 3
        df_states.loc[df_states["studyWave"] == 3, "close_interactions_raw"] = (
            df_states.loc[df_states["studyWave"] == 3, "selection_partners_01"].replace(
                wave3_mapping
            )
        )
        df_states.loc[
            (df_states["studyWave"] == 3) & (df_states["close_interactions_raw"] == -1),
            "close_interactions_raw",
        ] = np.nan

        interaction_stats = df_states.groupby(self.raw_esm_id_col)[
            "close_interactions_raw"
        ].apply(lambda x: x.sum() / x.count() if x.count() > 0 else np.nan)

        # Wave 2
        df_states.loc[df_states["studyWave"] == 2, "close_interactions"] = np.nan

        df_states["close_interactions"] = df_states[self.raw_esm_id_col].map(
            interaction_stats
        )
        self.close_interactions = deepcopy(
            df_states[["close_interactions", self.raw_esm_id_col]].drop_duplicates(
                keep="first"
            )
        )

        return df_states

    def create_conversation_topic_columns(
        self, df_states: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Creates columns for conversation topics:
            - work_conversations
            - personal_conversations
            - societal_conversations

        The assessment of these topics differs between study waves:
        - **Wave 1**: All categories are assessed, allowing calculation of percentages.
        - **Wave 2 & 3**: Only societal conversations are assessed; values for work and personal conversations are set to NaN.

        Processing logic:
        - For Wave 1, conversation topics are derived using binary masks.
        - Percentages of each conversation type are calculated per person for Wave 1.
        - For Wave 2 & 3, values for all conversation types are set to NaN.

        Args:
            df_states: The DataFrame containing interaction data.

        Returns:
            pd.DataFrame: The updated DataFrame with columns for conversation topics and their percentages.
        """
        conv_topic_vars = self.config_parser(
            self.cfg_preprocessing["esm_based"]["self_reported_micro_context"],
            "percentage",
            "work_conversations",
            "personal_conversations",
            "societal_conversations",
        )
        conversation_configs = {
            "work_conversations": {
                "cols": conv_topic_vars[0]["special_mappings"]["cocoms"]["wave1"][
                    "work_columns"
                ],
                "other_cols": conv_topic_vars[0]["special_mappings"]["cocoms"]["wave1"][
                    "other_columns"
                ],
            },
            "personal_conversations": {
                "cols": conv_topic_vars[1]["special_mappings"]["cocoms"]["wave1"][
                    "pers_columns"
                ],
                "other_cols": conv_topic_vars[1]["special_mappings"]["cocoms"]["wave1"][
                    "other_columns"
                ],
            },
            "societal_conversations": {
                "cols": conv_topic_vars[2]["special_mappings"]["cocoms"]["wave1"][
                    "soc_columns"
                ],
                "other_cols": conv_topic_vars[2]["special_mappings"]["cocoms"]["wave1"][
                    "other_columns"
                ],
            },
        }

        for conversation_type, config in conversation_configs.items():
            topic_mask = (df_states[config["cols"]].max(axis=1) == 1) & (
                df_states[config["other_cols"]].max(axis=1) == 0
            )
            other_mask = (df_states[config["other_cols"]].max(axis=1) == 1) & (
                df_states[config["cols"]].max(axis=1) == 0
            )
            df_states[conversation_type] = np.where(
                topic_mask, 1, np.where(other_mask, 0, np.nan)
            )

        df_states.loc[
            df_states["studyWave"] != 1,
            ["work_conversations", "personal_conversations", "societal_conversations"],
        ] = np.nan

        for col in [
            "work_conversations",
            "personal_conversations",
            "societal_conversations",
        ]:
            interaction_stats = df_states.groupby(self.raw_esm_id_col)[col].apply(
                lambda x: x.sum() / x.count() if x.count() > 0 else np.nan
            )
            df_states[col] = df_states[self.raw_esm_id_col].map(interaction_stats)

        self.work_conversations = deepcopy(
            df_states[["work_conversations", self.raw_esm_id_col]].drop_duplicates(
                keep="first"
            )
        )
        self.personal_conversations = deepcopy(
            df_states[["personal_conversations", self.raw_esm_id_col]].drop_duplicates(
                keep="first"
            )
        )
        self.societal_conversations = deepcopy(
            df_states[["societal_conversations", self.raw_esm_id_col]].drop_duplicates(
                keep="first"
            )
        )

        return df_states

    def create_relationship(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Infers the relationship status from the ESM surveys based on interactions with a partner.
        If a person interacted with their partner in StudyWave 1 or 3, all rows for that person will
        indicate a relationship. Rows corresponding to StudyWave 2 will be set to NaN.

        Processing logic:
        - **Wave 1 & 3**: Uses specific columns and partner interaction values to infer relationship status.
        - **Wave 2**: Relationship status is set to NaN by default.
        - The relationship status is inferred per person and applied across all rows for that person.

        Args:
            df_states: The DataFrame containing ESM data with interaction information.

        Returns:
            pd.DataFrame: The modified DataFrame with an added 'relationship' column indicating inferred relationship status.
        """

        relationship_cfg = self.config_parser(
            self.cfg_preprocessing["esm_based"]["self_reported_micro_context"],
            "binary",
            "relationship",
        )[0]
        waves_config = {
            1: {
                "column": relationship_cfg["item_names"][self.dataset]["wave1"],
                "partner_value": relationship_cfg["special_mappings"][self.dataset][
                    "wave1"
                ],
            },
            3: {
                "column": relationship_cfg["item_names"][self.dataset]["wave3"],
                "partner_value": relationship_cfg["special_mappings"][self.dataset][
                    "wave3"
                ],
            },
        }

        df_states["relationship"] = np.nan

        for wave, config in waves_config.items():
            ia_partner_col = config["column"]
            ia_partner_val = config["partner_value"]

            if ia_partner_col in df_states.columns:
                partner_interaction = df_states.groupby(self.raw_esm_id_col)[
                    ia_partner_col
                ].agg(lambda x: (x == ia_partner_val).any())
                df_states.loc[df_states["studyWave"] == wave, "relationship"] = (
                    df_states[self.raw_esm_id_col].map(partner_interaction).astype(int)
                )

            else:
                raise KeyError(f"Column {ia_partner_col} not in {self.dataset}")

        self.relationship = deepcopy(
            df_states[["relationship", self.raw_esm_id_col]].drop_duplicates(
                keep="first"
            )
        )

        return df_states

    def adjust_number_interactions_col(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts the 'selection_medium_00' column based on the time difference between the reported time of
        social interaction ('time_social_interaction') and the survey start time ('self.esm_timestamp'). If the
        interaction occurred more than one hour before the survey, the corresponding value in
        'selection_medium_00' is set to 0.

        Processing steps:
        - Ensures 'questionnaireStartedTimestamp' is in datetime format.
        - Converts 'time_social_interaction' from fractional hours to timedelta.
        - Adjusts 'selection_medium_00' for interactions that occurred more than 1 hour prior to the survey.

        Args:
            df_states: DataFrame containing the following columns:
                - 'questionnaireStartedTimestamp': The timestamp when the questionnaire was started.
                - 'time_social_interaction': Time of interaction in fractional hours (e.g., 16.75 = 16:45).
                - 'selection_medium_00': Binary column (1 for interaction, 0 for no interaction).

        Returns:
            pd.DataFrame: The adjusted DataFrame with updated 'selection_medium_00' column.
        """
        df_states[self.esm_timestamp] = df_states[self.esm_timestamp].apply(
            lambda x: x.split(".")[0] if isinstance(x, str) else x
        )
        df_states[f"{self.esm_timestamp}_dt"] = pd.to_datetime(
            df_states[self.esm_timestamp]
        )

        df_states["esm_timedelta"] = (
            df_states[f"{self.esm_timestamp}_dt"].dt.hour * 3600
            + df_states[f"{self.esm_timestamp}_dt"].dt.minute * 60
            + df_states[f"{self.esm_timestamp}_dt"].dt.second
        )
        df_states["esm_timedelta"] = pd.to_timedelta(
            df_states["esm_timedelta"], unit="s"
        )
        df_states["interaction_time"] = df_states["time_social_interaction"].apply(
            lambda x: (
                timedelta(hours=int(x), minutes=int((x % 1) * 60))
                if pd.notna(x)
                else pd.NaT
            )
        )

        df_states["selection_medium_00"] = df_states.apply(
            lambda row: (
                1
                if (row["esm_timedelta"] - row["interaction_time"]) > timedelta(hours=1)
                else row["selection_medium_00"]
            ),
            axis=1,
        )

        return df_states

    def dataset_specific_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs dataset-specific post-processing for CoCoMS. This includes merging additional derived
        variables (e.g., relationship status, interaction metrics, and conversation topic percentages)
        back into the main DataFrame.

        Args:
            df: The main DataFrame after general preprocessing steps.

        Returns:
            pd.DataFrame: The updated DataFrame with merged dataset-specific variables.
        """
        df = df.merge(self.relationship, on=self.raw_esm_id_col, how="left")
        df = df.merge(self.close_interactions, on=self.raw_esm_id_col, how="left")
        df = df.merge(self.work_conversations, on=self.raw_esm_id_col, how="left")
        df = df.merge(self.personal_conversations, on=self.raw_esm_id_col, how="left")
        df = df.merge(self.societal_conversations, on=self.raw_esm_id_col, how="left")

        return df

    def dataset_specific_sensing_processing(
        self, df_sensing: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Performs dataset-specific processing of sensing data for CoCoMS. This includes adjustments to
        the day cut-off times as the time is coded differently compared to zpid.

        Args:
            df_sensing: The DataFrame containing sensing data.

        Returns:
            pd.DataFrame: The updated DataFrame with adjustments applied.
        """
        df_sensing = self.adjust_day_cut_off(df_sensing=df_sensing)
        return df_sensing

    @staticmethod
    def adjust_day_cut_off(df_sensing: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts the day cut-off for sensing variables to 3 AM. This ensures that late-night events
        (e.g., smartphone usage) are assigned to the correct day for analysis.

        Specifically, it:
        - Shifts events occurring before 3:00 AM to the previous day by adding 24 hours.
        - Adjusts all times by subtracting 3 hours to align the day cut-off at 3 AM.
        - Converts the adjusted times to minutes.

        Args:
            df_sensing: Input DataFrame containing 'Screen_firstEvent' and 'Screen_lastEvent' columns.

        Returns:
            pd.DataFrame: Updated DataFrame with adjusted 'Screen_firstEvent' and 'Screen_lastEvent' columns.
        """
        df_sensing["Screen_firstEvent"] = df_sensing["Screen_firstEvent"].apply(
            lambda x: x + 24 if x < 3 else x
        )
        df_sensing["Screen_lastEvent"] = df_sensing["Screen_lastEvent"].apply(
            lambda x: x + 24 if x < 3 else x
        )

        df_sensing["Screen_firstEvent"] = (df_sensing["Screen_firstEvent"] - 3) * 60
        df_sensing["Screen_lastEvent"] = (df_sensing["Screen_lastEvent"] - 3) * 60

        return df_sensing
