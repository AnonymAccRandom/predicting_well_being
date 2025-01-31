import numpy as np
import pandas as pd

from src.preprocessing.BasePreprocessor import BasePreprocessor
from src.utils.utilfuncs import NestedDict


class CocoutPreprocessor(BasePreprocessor):
    """
    Preprocessor for the "cocout" dataset, inheriting from BasePreprocessor.

    This class implements preprocessing logic specific to the "cocout" dataset.
    It inherits all the attributes and methods of BasePreprocessor, including:
    - Configuration file ('cfg_preprocessing').
    - Logging and timing utilities (`logger`, `timer`).
    - Data loading, processing, and sanity checking methods.

    Additional Attributes (for other Attributes, see BasePreprocessor):
        dataset (str): Specifies the current dataset as "cocout".
    """

    def __init__(self, cfg_preprocessing: NestedDict) -> None:
        """
        Initializes the CocoutPreprocessor with dataset-specific configurations.

        Args:
            cfg_preprocessing: Yaml config specifying details on preprocessing (e.g., scales, items).
        """
        super().__init__(cfg_preprocessing=cfg_preprocessing)
        self.dataset = "cocout"

    def merge_traits(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges trait-related DataFrames from multiple datasets (ut1 and ut2) and handles inconsistencies in column names.

        For each dataset:
        - Adds suffixes to distinguish overlapping column names across data types (e.g., `_bfi`, `_se`, `_dem`).
        - Drops rows with missing participant IDs in key columns.
        - Resolves column name inconsistencies specific to the dataset using `fix_col_name_issues`.
        - Merges trait-related DataFrames (`data_traits`, `data_personality`, `data_demographics`, and `data_self_esteem`)
          along the columns using the participant ID as the key.

        The final DataFrame is created by concatenating the processed DataFrames for `ut1` and `ut2` along the rows.

        Args:
            df_dct: Dictionary containing the input DataFrames for each dataset and data type.

        Returns:
            pd.DataFrame: A combined DataFrame containing trait-level information for all datasets.
        """
        dataset_lst = []
        for dataset in ["ut1", "ut2"]:
            df_dct[f"data_personality_{dataset}"] = df_dct[
                f"data_personality_{dataset}"
            ].add_suffix("_bfi")
            df_dct[f"data_personality_{dataset}"] = df_dct[
                f"data_personality_{dataset}"
            ].dropna(subset="pID_bfi")
            df_dct[f"data_self_esteem_{dataset}"] = df_dct[
                f"data_self_esteem_{dataset}"
            ].add_suffix("_se")
            df_dct[f"data_self_esteem_{dataset}"] = df_dct[
                f"data_self_esteem_{dataset}"
            ].dropna(subset="pID_se")
            df_dct[f"data_demographics_{dataset}"] = df_dct[
                f"data_demographics_{dataset}"
            ].add_suffix("_dem")
            df_dct[f"data_demographics_{dataset}"] = df_dct[
                f"data_demographics_{dataset}"
            ].dropna(subset="pID_dem")

            if dataset == "ut2":
                self.fix_col_name_issues(df_dct)

            df_traits = (
                df_dct[f"data_traits_t1_{dataset}"]
                .merge(
                    df_dct[f"data_traits_t2_{dataset}"],
                    left_on="id",
                    right_on="id",
                    how="outer",
                    validate="1:1",
                    suffixes=(None, "_postsurv"),
                )
                .dropna(subset="id")
                .reset_index(drop=True)
                .merge(
                    df_dct[f"data_personality_{dataset}"],
                    left_on="id",
                    right_on="pID_bfi",
                    how="outer",
                    validate="1:1",
                )
                .dropna(subset="id")
                .reset_index(drop=True)
                .merge(
                    df_dct[f"data_demographics_{dataset}"],
                    left_on="id",
                    right_on="pID_dem",
                    how="outer",
                    validate="1:m",
                )
                .dropna(subset="id")
                .reset_index(drop=True)
                .merge(
                    df_dct[f"data_self_esteem_{dataset}"],
                    left_on="id",
                    right_on="pID_se",
                    how="outer",
                    validate="1:1",
                )
                .dropna(subset="id")
                .reset_index(drop=True)
            )
            dataset_lst.append(df_traits)

        concatenated_traits = pd.concat(dataset_lst, axis=0)

        return concatenated_traits

    @staticmethod
    def fix_col_name_issues(df_dct: dict[str, pd.DataFrame]) -> None:
        """
        Resolves column name inconsistencies between CoCo UT1 and CoCo UT2 datasets.

        For CoCo UT2:
        - Renames columns in `data_traits_t1` and `data_traits_t2` that contain the prefix "cms_" to "cmq_".

        Note: This modifies the dict inplace, so nothing is returned. The changes are made directly to the input dict.

        Args:
            df_dct: Dictionary containing DataFrames for different datasets and data types.
        """
        dataset = "ut2"

        df_t1 = df_dct[f"data_traits_t1_{dataset}"]
        df_t2 = df_dct[f"data_traits_t2_{dataset}"]

        cmq_cols = [col for col in df_t1.columns if "cms_" in col]

        df_t1 = df_t1.rename(
            columns={col: col.replace("cms_", "cmq_") for col in cmq_cols}
        )
        df_t2 = df_t2.rename(
            columns={col: col.replace("cms_", "cmq_") for col in cmq_cols}
        )

        df_dct[f"data_traits_t1_{dataset}"] = df_t1
        df_dct[f"data_traits_t2_{dataset}"] = df_t2

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans column names in a DataFrame by removing a specified suffix and adjusting regex patterns.

        Specifically:
        - Removes the "_t1" suffix from column names if present.
        - Updates the DataFrame with the cleaned column names.

        Args:
            df_traits: A pandas DataFrame containing columns that may have duplicates or unnecessary suffixes.

        Returns:
            pd.DataFrame: The modified DataFrame with cleaned column names.
        """
        trait_suffix = self.cfg_preprocessing["general"]["pl_suffixes"]["cocout"]
        updated_columns = []

        for col in df_traits.columns:
            if col.endswith(trait_suffix):
                col = col[: -len(trait_suffix)]
            updated_columns.append(col)

        df_traits.columns = updated_columns
        return df_traits

    def dataset_specific_trait_processing(
        self, df_traits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes trait data specific to CoCo UT by creating and populating new columns:
        - "professional_status": Assigned a constant value of 1.
        - "educational_attainment": Assigned a constant value of 4 (representing 'Abi equivalent' on a 1-6 scale).
        - "studyWave": Initialized with NaN.
        - "country": Assigned a constant value of "usa".

        Args:
            df_traits: The input DataFrame containing trait data.

        Returns:
            pd.DataFrame: The modified DataFrame with the new columns added.
        """
        df_traits["professional_status"] = 1
        df_traits["educational_attainment"] = 4
        df_traits["studyWave"] = np.nan
        df_traits["country"] = "usa"

        return df_traits

    def merge_states(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges state-level ESM and ESM daily DataFrames for CoCo UT datasets (ut1 and ut2) by aligning them
        on participant IDs and date. The merged DataFrames are then concatenated across datasets.

        Args:
            df_dct: A dictionary containing the state-level DataFrames for both "ut1" and "ut2" datasets.

        Returns:
            pd.DataFrame: A single concatenated DataFrame containing the merged state-level data for all datasets.
        """
        df_lst = []

        for dataset in ["ut1", "ut2"]:
            data_esm = df_dct[f"data_esm_{dataset}"]
            data_esm_daily = df_dct[f"data_esm_daily_{dataset}"]
            data_esm["date"] = pd.to_datetime(data_esm["RecordedDateConvert"]).dt.date
            data_esm_daily["date"] = pd.to_datetime(
                data_esm_daily["RecordedDateConvert"]
            ).dt.date

            merged_df = pd.merge(
                data_esm,
                data_esm_daily,
                how="left",
                on=["id", "date"],
                suffixes=("_esm", "_daily"),
            )
            df_lst.append(merged_df)

        concatenated_esm = pd.concat(df_lst, axis=0).reset_index(drop=True)

        return concatenated_esm

    def dataset_specific_state_processing(
        self, df_states: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Applies dataset-specific processing to the state-level data for CoCo UT datasets.

        Adjustments include:
        - Cleaning and processing the 'number_interaction_partners' column using the `clean_number_interaction_partners` method.
        - Extracting and assigning the study wave information from the participant ID.

        Args:
            df_states: A DataFrame containing the state-level data.

        Returns:
            pd.DataFrame: The modified state-level DataFrame with dataset-specific adjustments.
        """
        df_states = self.clean_number_interaction_partners(df_states=df_states)
        df_states["studyWave"] = df_states[self.raw_trait_id_col].apply(lambda x: x[:4])

        return df_states

    def clean_number_interaction_partners(
        self, df_states: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Cleans the 'number_interaction_partners' column for CoCo UT datasets by:
        - Converting unambiguous strings (e.g., "one") to numerical values.
        - Setting ambiguous or invalid strings to `np.nan`.

        Args:
            df_states: A DataFrame containing the state-level data.

        Returns:
            pd.DataFrame: The modified DataFrame with a cleaned 'number_interaction_partners' column.
        """
        number_int_partners_cfg = [
            entry
            for entry in self.cfg_preprocessing["esm_based"][
                "self_reported_micro_context"
            ]
            if entry["name"] == "number_interaction_partners"
        ][0]
        col_name = number_int_partners_cfg["item_names"]["cocout"]

        df_states[col_name] = df_states[col_name].replace(
            number_int_partners_cfg["special_mappings"]["cocout"]["str_to_num"]
        )
        df_states[col_name] = pd.to_numeric(df_states[col_name], errors="coerce")

        return df_states

    def dataset_specific_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies post-processing steps specific to the CoCo UT dataset. We assign zero to the 'full_employed' and
        "unemployed" columns as this is a student sample.

        Args:
            df: A DataFrame containing the processed data.

        Returns:
            pd.DataFrame: The modified DataFrame with the additional columns.
        """
        df["full_employed"] = 0
        df["unemployed"] = 0

        return df
