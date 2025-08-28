from copy import deepcopy
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from src.preprocessing.BasePreprocessor import BasePreprocessor
from src.utils.utilfuncs import NestedDict


class ZpidPreprocessor(BasePreprocessor):
    """
    Preprocessor for the "zpid" dataset, inheriting from BasePreprocessor.

    This class implements preprocessing logic specific to the "zpid" dataset.
    It inherits all the attributes and methods of BasePreprocessor, including:
    - Configuration file ('cfg_preprocessing').
    - Logging and timing utilities (`logger`, `timer`).
    - Data loading, processing, and sanity checking methods.

    Additional Attributes (for other Attributes, see BasePreprocessor):
        dataset (str): Specifies the current dataset as "zpid".
        home_office (Any): Stores data related to home office, assigned during preprocessing.
    """

    def __init__(self, cfg_preprocessing: NestedDict) -> None:
        """
        Initializes the ZpidPreprocessor with dataset-specific configurations.

        Args:
            cfg_preprocessing: Yaml config specifying details on preprocessing (e.g., scales, items).
        """
        super().__init__(cfg_preprocessing=cfg_preprocessing)
        self.dataset = "zpid"
        self.home_office = None

    def merge_traits(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Retrieves the DataFrame for trait-level data from the input dictionary.

        Args:
            df_dct: A dictionary containing multiple DataFrames, with keys indicating their types.

        Returns:
            pd.DataFrame: The DataFrame containing trait-level data.
        """
        return df_dct["data_traits"]

    def merge_states(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Retrieves the DataFrame for state-level data from the input dictionary.

        Args:
            df_dct: A dictionary containing multiple DataFrames, with keys indicating their types.

        Returns:
            pd.DataFrame: The DataFrame containing state-level data.
        """
        return df_dct["data_esm"]

    def dataset_specific_trait_processing(
        self, df_traits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Applies ZPID-specific processing to trait-level data, including handling missing values,
        adjusting professional status, and creating additional columns.

        Steps:
        - Replaces specific values with NaN based on configuration.
        - Merges BFI items from multiple waves to improve reliability.
        - Adjusts professional status based on working days per week.
        - Creates a home office percentage column.
        - Assign the sample of the country and a time column to the DataFrame.

        Args:
            df_traits: The DataFrame containing trait-level data.

        Returns:
            pd.DataFrame: The updated DataFrame with processed trait-level data.
        """
        df_traits = self.replace_values(df_traits=df_traits)
        df_traits = self.merge_bfi_items(df_traits=df_traits)
        df_traits = self.adjust_professional_status_col(df_traits=df_traits)
        df_traits = self.create_home_office(df_traits=df_traits)

        df_traits["created_demog"] = "2020-01-01"
        df_traits["country"] = "germany"

        return df_traits

    @staticmethod
    def replace_values(df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces specific values in the trait-level data with NaN based on configuration.

        This method replaces certain value that are placeholders for missing data or need to be set to
        np.nan for other reasons (e.g., invalid values).

        Args:
            df_traits: The DataFrame containing trait-level data.

        Returns:
            pd.DataFrame: The updated DataFrame with replaced values.
        """
        df_traits = df_traits.replace(-77, np.nan)

        df_traits.loc[df_traits["Demo_GL2"] == 5, "Demo_GL2"] = np.nan
        df_traits.loc[df_traits["Polit_Ein_4"] == 12, "Polit_Ein_4"] = np.nan
        df_traits.loc[df_traits["Demo_B1"] == 0, "Demo_B1"] = np.nan
        df_traits.loc[
            df_traits["Arbeitsleben91_AugOct"] == 0, "Arbeitsleben91_AugOct"
        ] = np.nan
        df_traits.loc[df_traits["Demo_AL1"] == 0, "Demo_AL1"] = np.nan

        return df_traits

    def merge_bfi_items(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Merges BFI2XS items assessed in two waves to compensate for missing values and increase reliability.

        **Processing logic**:
        - Sanity check if the values of the BFI_2XS waves corresponds to the same scale.
        - For items with values in both waves, the mean is calculated.
        - For items with values in one wave, the available value is used.
        - Items without any values are set to NaN.
        - Values outside the scale boundaries are set to NaN.

        Args:
            df_traits: The DataFrame containing BFI items across multiple waves.

        Returns:
            pd.DataFrame: The updated DataFrame with merged BFI items.
        """
        self.logger.log("        Check if params of the BFI_2XS waves are comparable")
        bfi_items = set(
            [col.rsplit("_", 1)[0] for col in df_traits.columns if "BFI_2XS" in col]
        )

        for item in bfi_items:
            for wave in ["wave3", "wave4"]:
                col_name = f"{item}_{wave}"
                if col_name in df_traits.columns:
                    stats = df_traits[col_name].describe()
                    self.logger.log(
                        f"          Stats for {col_name} - "
                        f"            M: {round(stats['mean'], 3)}, "
                        f"            SD: {round(stats['std'], 3)}, "
                        f"            Min: {round(stats['min'], 3)}, "
                        f"            Max: {round(stats['max'], 3)}"
                    )
                else:
                    self.logger.log(
                        f"WARNING: Column {col_name} not found in DataFrame."
                    )

            wave3_col = f"{item}_wave3"
            wave4_col = f"{item}_wave4"
            df_traits[wave3_col] = df_traits[wave3_col].where(
                df_traits[wave3_col].between(1, 5), other=np.nan
            )
            df_traits[wave4_col] = df_traits[wave4_col].where(
                df_traits[wave4_col].between(1, 5), other=np.nan
            )

            df_traits[item] = df_traits[[wave3_col, wave4_col]].mean(
                axis=1, skipna=True
            )

        self.logger.log(f"        Removed values that are not between 1 and 5")
        df_traits.drop(
            columns=[f"{item}_wave3" for item in bfi_items]
            + [f"{item}_wave4" for item in bfi_items],
            inplace=True,
        )

        return df_traits

    @staticmethod
    def adjust_professional_status_col(df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts the professional status column "Demo_AL1".

        It changes the categories 1 (Angestellter) and
        3 (Beamter) to a new category (15) if the number of days worked per week in the "MCTQ_WD_2" column
        is less than 5.

        Args:
            df_traits: The DataFrame containing trait-level data, including columns "Demo_AL1"
                       (professional status) and "MCTQ_WD_2" (working days per week).

        Returns:
            pd.DataFrame: The updated DataFrame with adjusted "Demo_AL1" values where applicable.
        """
        condition = (df_traits["MCTQ_WD_2"] < 5) & df_traits["Demo_AL1"].isin([1, 3])
        df_traits.loc[condition, "Demo_AL1"] = 14

        return df_traits

    def create_home_office(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a "home_office" column as a percentage of the total days worked from home.

        **Processing logic**:
        - Maps the original categorical values of home office data to numerical values.
        - Divides the mapped value by 7 to calculate the percentage of working days spent at home.

        Args:
            df_traits: The DataFrame containing trait-level data, including the home office column.

        Returns:
            pd.DataFrame: The updated DataFrame with a "home_office" column.
        """
        home_office_cfg = self.config_parser(
            self.cfg_preprocessing["person_level"]["sociodemographics"],
            "percentage",
            "home_office",
        )[0]
        col_name = home_office_cfg["item_names"][self.dataset]

        df_traits[col_name] = df_traits[col_name].map(
            home_office_cfg["category_mappings"][self.dataset]
        )
        df_traits["home_office"] = df_traits[col_name] / 7  # 7 days would be 100%

        self.home_office = deepcopy(
            df_traits[["home_office", self.raw_trait_id_col]].drop_duplicates(
                keep="first"
            )
        )

        return df_traits

    def dataset_specific_state_processing(
        self, df_states: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes state-level data for ZPID by adding a "studyWave" column to track participation in study waves.

        Args:
            df_states: The DataFrame containing state-level data.

        Returns:
            pd.DataFrame: The updated DataFrame with the "studyWave" column.
        """
        df_states = self.create_wave_col(df_states)
        df_states["country"] = "germany"
        return df_states

    def create_wave_col(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a "studyWave" column to indicate participants' involvement in different study waves.

        **Processing logic**:
        - Wave 1: Participation is identified if the earliest timestamp is before or on August 31, 2020.
        - Wave 2: Participation is identified if the latest timestamp is on or after September 1, 2020.
        - Both: Indicates participation in both Wave 1 and Wave 2.

        Args:
            df_states: The DataFrame containing ESM data, including a timestamp column (`self.esm_timestamp`)
                       and participant IDs (`self.raw_esm_id_col`).

        Returns:
            pd.DataFrame: The input DataFrame augmented with a new "studyWave" column that tracks wave participation.
        """
        wave_1_end = datetime(year=2020, month=8, day=31)
        wave_2_start = datetime(year=2020, month=9, day=1)

        df_states[self.esm_timestamp] = pd.to_datetime(df_states[self.esm_timestamp])
        grouped = (
            df_states.groupby(self.raw_esm_id_col)[self.esm_timestamp]
            .agg(["min", "max"])
            .reset_index()
        )

        def assign_wave(row: pd.Series) -> Union[np.nan, int, str]:
            """
            Assigns the wave participation status for each participant based on their earliest and latest timestamps.

            Args:
                row: A pandas Series representing a participant's earliest ('min') and latest ('max') timestamps.

            Returns:
                Union[np.nan, int, str]: "Both" if participated in both waves, 1 if only in Wave 1,
                                         2 if only in Wave 2, and np.nan if no valid participation.
            """
            first_timestamp = row["min"]
            last_timestamp = row["max"]

            participated_in_wave_1 = first_timestamp <= wave_1_end
            participated_in_wave_2 = last_timestamp >= wave_2_start

            if participated_in_wave_1 and participated_in_wave_2:
                return "Both"

            elif participated_in_wave_1:
                return 1

            elif participated_in_wave_2:
                return 2

            return np.nan

        grouped["studyWave"] = grouped.apply(assign_wave, axis=1)
        df_states = df_states.merge(
            grouped[[self.raw_esm_id_col, "studyWave"]],
            on=self.raw_esm_id_col,
            how="left",
        )

        return df_states

    def dataset_specific_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds home office data to the main DataFrame during the post-processing stage.

        Args:
            df: The DataFrame containing the preprocessed data.

        Returns:
            pd.DataFrame: The updated DataFrame with merged home office data.
        """
        df = df.merge(self.home_office, on=self.raw_trait_id_col, how="left")
        return df
