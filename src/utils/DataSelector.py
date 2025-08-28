import os
import sys
from copy import copy

import pandas as pd

from src.utils.DataLoader import DataLoader
from src.utils.utilfuncs import NestedDict


class DataSelector:
    """
    A class responsible for selecting data based on the specified combination of features and samples.

    This class processes a full DataFrame by:
    - Filtering rows based on the desired sample inclusion criteria.
    - Filtering columns based on the specified feature combination.
    - Extracting the criterion variable for analysis.
    - Optionally selecting the best features based on prior results.

    Attributes:
        cfg_analysis (NestedDict): Yaml config specifying details on the ML analysis (e.g., CV, models).
        df (pd.DataFrame): Input DataFrame containing all datasets, features, and metadata columns.
        feature_combination (str): Feature combination to use for analysis (e.g., "pl_srmc_mac").
        crit (str): Criterion variable to predict (e.g., "wb_state").
        samples_to_include (str): Sample inclusion criteria (e.g., "selected", "control", "all").
        meta_vars (list[str]): List of metadata columns needed for downstream processing.
        datasets_included (list[str] | None): Datasets included based on the criteria.
        X (pd.DataFrame | None): Selected features DataFrame.
        y (pd.Series | None): Selected criterion variable.
    """

    def __init__(
        self,
        cfg_analysis: NestedDict,
        df: pd.DataFrame,
        feature_combination: str,
        crit: str,
        samples_to_include: str,
        meta_vars: list[str],
    ):
        """
        Initializes the DataSelector with configuration and input data.

        Args:
            cfg_analysis: Yaml config specifying details on the ML analysis (e.g., CV, models).
            df: DataFrame containing all datasets, features, and metadata columns.
            feature_combination: Feature combination to use for analysis (e.g., "pl_srmc_mac").
            crit: Criterion variable to predict (e.g., "wb_state").
            samples_to_include: Sample inclusion criteria (e.g., "selected", "control", "all").
            meta_vars: List of metadata columns required for downstream processing.
        """
        self.cfg_analysis = cfg_analysis
        self.feature_combination = feature_combination
        self.crit = crit
        self.samples_to_include = samples_to_include
        self.meta_vars = meta_vars
        self.datasets_included = None

        self.df = df
        self.X = None
        self.y = None

        self.data_loader = DataLoader()

    def select_samples(self) -> None:
        """
        Selects samples based on the specified `samples_to_include` and `feature_combination`.

        - For "selected" and "control", filters datasets to reduce NaNs and include only relevant samples.
        - For "all", includes all datasets independent of features used for the analysis.
        - Excludes rows with missing values in the criterion column.
        - Optionally samples rows for testing if specified in the config.

        Some more specifics:
            - For the "sens" analysis where samples_to_include == selected, only samples with sensing data are included
            - Another analysis was added to fit certain features on a specific data subset (feature_combination -> _control,
            i.e., pl_control, pl_srmc_control)

        At the end of the function, it sets the filtered df as the class attribute.
        """
        if self.samples_to_include in ["selected", "control"]:
            datasets_included = self.cfg_analysis["feature_sample_combinations"][
                self.feature_combination
            ]
            datasets_included_filtered = [
                dataset
                for dataset in datasets_included
                if dataset in self.cfg_analysis["crit_available"][self.crit]
            ]
            self.datasets_included = datasets_included_filtered

            df_filtered = self.df[
                self.df.index.to_series().apply(
                    lambda x: any(
                        x.startswith(sample) for sample in self.datasets_included
                    )
                )
            ]

            if (
                self.samples_to_include in ["selected", "control"]
                and "sens" in self.feature_combination
            ):
                sens_columns = [
                    col for col in self.df.columns if col.startswith("sens_")
                ]
                df_filtered = df_filtered[df_filtered[sens_columns].notna().any(axis=1)]

            if "_control" in self.feature_combination:
                sens_columns = [
                    col for col in self.df.columns if col.startswith("sens_")
                ]
                df_filtered = df_filtered[df_filtered[sens_columns].notna().any(axis=1)]

        else:
            datasets_included_filtered = [
                dataset
                for dataset in self.cfg_analysis["feature_sample_combinations"][
                    "all_in"
                ]
                if dataset in self.cfg_analysis["crit_available"][self.crit]
            ]
            self.datasets_included = datasets_included_filtered

            df_filtered = self.df[
                self.df.index.to_series().apply(
                    lambda x: any(
                        x.startswith(sample) for sample in self.datasets_included
                    )
                )
            ]

        crit_col = f"crit_{self.crit}"
        df_filtered_crit_na = df_filtered.dropna(subset=[crit_col])

        if self.cfg_analysis["tests"]["sample"]:
            sample_size = self.cfg_analysis["tests"]["sample_size"]
            df_filtered_crit_na = df_filtered_crit_na.sample(
                n=sample_size, random_state=self.cfg_analysis["random_state"]
            )

        self.df = df_filtered_crit_na

    def select_features(self) -> pd.DataFrame:
        """
        Filters the columns of the DataFrame based on the feature combination.

        - Includes features with specific prefixes (e.g., "pl_", "srmc_", "mac_").
        - Adds metadata columns needed for downstream processing.
        - Handles special cases like removing certain features for specific analyses (_nnse, _npers)

        Returns:
            pd.DataFrame: DataFrame containing only the selected features.
        """
        selected_columns = copy(self.meta_vars)
        if self.samples_to_include in ["all", "selected"]:
            feature_prefix_lst = self.feature_combination.split("_")

            if "all_in" in self.feature_combination:  # include all features
                feature_prefix_lst = ["pl", "srmc", "sens", "mac"]
                if self.samples_to_include == "selected":
                    sys.exit(0)

        elif self.samples_to_include == "control":
            feature_prefix_lst = ["pl"]

            no_control_lst = self.cfg_analysis["no_control_lst"]
            if self.feature_combination in no_control_lst:
                sys.exit(0)

        else:
            raise ValueError(
                f"Invalid value #{self.samples_to_include}# for attr samples_to_include"
            )

        if self.feature_combination == "all":  # include all features
            feature_prefix_lst = ["pl", "srmc", "sens", "mac"]

        # always include grouping id
        for feature_cat in feature_prefix_lst:
            for col in self.df.columns:
                if col.startswith(feature_cat):
                    selected_columns.append(col)

        if "nnse" in self.feature_combination:
            to_remove = [
                "pl_depression",
                "pl_anxiety",
                "pl_emotional_volatility",
                "pl_self_esteem",
            ]
            selected_columns = [col for col in selected_columns if col not in to_remove]

        if "npers" in self.feature_combination:
            to_remove = self.cfg_analysis["pl_vars_to_exclude"]
            selected_columns = [col for col in selected_columns if col not in to_remove]

        X = self.df[selected_columns].copy()
        setattr(self, "X", X)

        return X

    def select_criterion(self) -> pd.Series:
        """
        Extracts the criterion column based on the analysis specification.

        Ensures the number of rows in the feature DataFrame matches the criterion.

        Returns:
            pd.Series: The criterion column.
        """
        y = self.df[f"crit_{self.crit}"]

        assert len(self.X) == len(
            y
        ), f"Features and criterion differ in length, len(X) == {len(self.X)}, len(y) == {len(y)}"

        setattr(self, "y", y)

        return y

    def select_best_features(
        self, df: pd.DataFrame, root_path: str, model: str, num_features: int = 10
    ) -> pd.DataFrame:
        """
        Filters the DataFrame to include only the top features based on prior feature selection results.

        Args:
            df: The DataFrame containing the data.
            root_path: Root directory containing feature selection results.
            model: Model type (e.g., "randomforestregressor").
            num_features: Number of top features to include. Defaults to 10.

        Note: This is only used in postprocessing, not in the ML-analysis.

        Returns:
            pd.DataFrame: Filtered DataFrame including only the top features.
        """
        file_path = os.path.join(
            root_path,
            self.feature_combination,
            self.samples_to_include,
            self.crit,
            model,
            f"top_{num_features}_features.txt",
        )

        feature_lst = self.data_loader.read_txt(file_path)
        df_filtered = df[feature_lst + self.meta_vars]

        return df_filtered
