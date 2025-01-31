import os
from collections import defaultdict

import numpy as np
import pandas as pd
import shap
from shap import Explanation

from src.utils.DataLoader import DataLoader
from src.utils.DataSaver import DataSaver
from src.utils.utilfuncs import apply_name_mapping, defaultdict_to_dict, NestedDict


class ShapProcessor:
    """
    Processes SHAP values for machine learning models.

    This class is responsible for:
    - Summarizing SHAP values obtained across outer folds of cross-validation.
    - Recreating SHAP explanation objects for plotting and analysis.

    Attributes:
        cfg_postprocessing: Configuration dictionary for postprocessing settings.
        name_mapping: Mapping of feature combination keys to display names.
        cv_shap_results_path: Path to the cross-validation SHAP results.
        shap_ia_values_path: Path to store interaction SHAP values.
        data_loader: Instance of `DataLoader` for loading data.
        data_saver: Instance of `DataSaver` for saving data.
        shap_values_file_name: Filename for summarized SHAP values.
        shap_ia_values_file_name: Filename for summarized interaction SHAP values.
        meta_vars: List of meta variables (e.g., identifiers or columns excluded from SHAP processing).
    """

    def __init__(
        self,
        cfg_postprocessing: NestedDict,
        name_mapping: NestedDict,
        base_result_path: str,
        cv_shap_results_path: str,
        processed_results_filenames: dict[str, str],
        meta_vars: list[str],
    ) -> None:
        """
        Initializes the ShapProcessor with configuration settings, file paths, and filenames.

        Args:
            cfg_postprocessing: Configuration dictionary containing postprocessing settings.
            name_mapping: A nested dictionary mapping feature combination keys to display names.
            base_result_path: The base directory path where results are stored.
            cv_shap_results_path: Path to the directory containing cross-validation SHAP results.
            processed_results_filenames: Dictionary containing filenames for processed results.
            meta_vars: List of meta variables to exclude from SHAP processing.
        """
        self.cfg_postprocessing = cfg_postprocessing
        self.name_mapping = name_mapping

        self.cv_shap_results_path = cv_shap_results_path
        self.shap_ia_values_path = os.path.join(
            base_result_path,
            self.cfg_postprocessing["general"]["data_paths"]["ia_values"],
        )

        self.data_loader = DataLoader()
        self.data_saver = DataSaver()
        self.shap_values_file_name = processed_results_filenames[
            "shap_values_summarized"
        ]
        self.shap_ia_values_file_name = processed_results_filenames[
            "shap_ia_values_summarized"
        ]

        self.meta_vars = meta_vars

    @classmethod
    def nested_dict(cls) -> defaultdict:
        """
        Creates a nested defaultdict that allows for the creation of arbitrarily deep dictionaries.

        This is useful for dynamically constructing nested dictionaries without having to explicitly
        initialize each level.

        Returns:
            defaultdict: A defaultdict where each level defaults to another nested defaultdict.
        """
        return defaultdict(cls.nested_dict)

    def prepare_shap_data(
        self,
        model_to_plot: str,
        crit_to_plot: str,
        samples_to_include: str,
        col_assignment: dict[str, list[str]],
    ) -> dict[str, Explanation]:
        """
        Prepares the data for SHAP visualization in the beeswarm plots.

        This function prepares the data for the SHAP beeswarm plots. It extracts the relevant SHAP values
        and stores them in a dict that is returned.
        In the final usage
        - we plot all feature_combinations for a given samples_to_include - crit - model combination
        - we always use "samples_to_include" == selected, expect for the all_in analysis

        Args:
            model_to_plot: The model name to filter by.
            crit_to_plot: The criterion name to filter by.
            samples_to_include: Default samples to include.
            col_assignment: Defined the position of a feature combination in the plot

        Returns:
            dict: Nested dictionary with SHAP explanation objects.
        """
        result_dct = self.nested_dict()

        for root, dirs, files in os.walk(self.cv_shap_results_path):
            if self.shap_values_file_name in files:
                relative_path = os.path.relpath(root, self.cv_shap_results_path)
                parts = relative_path.split(os.sep)

                if len(parts) == 4:
                    feature_combination, samples, crit, model = parts

                    if (
                        samples_to_include == "combo"
                    ):  # currently not used in the SHAP plots
                        required_samples = self.get_required_sample_for_combo(
                            feature_combination=feature_combination,
                            col_assignment=col_assignment,
                        )
                    else:
                        required_samples = samples_to_include

                    if feature_combination == "all_in":
                        required_samples = "all"

                    # Only load if all filters match
                    if (
                        crit == crit_to_plot
                        and samples == required_samples
                        and model == model_to_plot
                    ):
                        shap_values_path = os.path.join(
                            root, self.shap_values_file_name
                        )
                        shap_values = self.data_loader.read_pkl(shap_values_path)

                        feature_names_raw = [
                            feature
                            for feature in shap_values["feature_names"]
                            if feature not in self.meta_vars
                        ]
                        feature_names_formatted = apply_name_mapping(
                            features=feature_names_raw,
                            name_mapping=self.name_mapping,
                            prefix=True,
                        )

                        # Recreate explanation objects
                        shap_exp = self.recreate_shap_exp_objects(
                            shap_values=np.array(shap_values["shap_values"]["mean"]),
                            base_values=np.array(shap_values["base_values"]["mean"]),
                            data=np.array(shap_values["data"]["mean"]),
                            feature_names=feature_names_formatted,
                        )
                        result_dct[feature_combination] = shap_exp

                        # Store most important features
                        self.get_most_important_features(
                            shap_values=shap_exp.values,
                            feature_names=feature_names_raw,
                            root=root,
                            n=self.cfg_postprocessing["calculate_exp_lin_models"][
                                "num_features"
                            ],
                            store=self.cfg_postprocessing["calculate_exp_lin_models"][
                                "store"
                            ],
                        )
        return defaultdict_to_dict(result_dct)

    def get_most_important_features(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        root: str,
        n: int,
        store: bool = True,
    ) -> None:
        """
        Extracts the x most important features for a given analysis setting.

        Args:
            shap_values: ndarray containing the SHAP values for a specific analysis.
            feature_names: List containing the feature names.
            root: Directory path of the current analysis where the output file will be stored.
            n: int, number of top features to select.
            store: If true, store the feature_names in "root"
        """
        mean_abs_values = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(-mean_abs_values)[:n]
        top_features = [feature_names[i] for i in top_indices]

        if store:
            output_file = os.path.join(root, f"top_{n}_features.txt")
            self.data_saver.save_txt(output_file, top_features)

    def prepare_shap_ia_data(
        self,
        model_to_plot: str,
        crit_to_plot: str,
        samples_to_include: str,
        feature_combination_to_plot: str,
    ) -> dict[str, Explanation]:
        """
        Prepares the SHAP interaction data for visualization.

        This method filters and processes SHAP interaction values based on the specified model, criterion,
        sample inclusion, and feature combination. It reconstructs SHAP explanation objects suitable for
        visualization (e.g., beeswarm plots) by combining the interaction effects of features.
            - The method traverses directories within the specified SHAP interaction values path, identifying
              files matching the expected structure and filtering by the given parameters.
            - SHAP interaction values are loaded, processed, and formatted:
                - Interaction values of the best 20 feature / feature pairs are extracted (for the complete dataset)
                - Values are sorted by their absolute mean interaction effect.
                - Feature pairs are formatted using `apply_name_mapping`.
            - Data for the top N interactions is retrieved and used to reconstruct a SHAP explanation object.
            - The SHAP explanation object is stored in the result dictionary.

        Args:
            model_to_plot: The name of the model to filter data by.
            crit_to_plot: The criterion name to filter data by.
            samples_to_include: The sample inclusion criteria.
            feature_combination_to_plot: The name of the feature combination to filter data by

        Returns:
            dict: A nested dictionary where each key corresponds to a feature combination and the value is
            a SHAP explanation object. These objects contain:
                - SHAP interaction values for top feature interactions.
                - Associated metadata (e.g., base values, feature names).
                - Data for creating visualizations.
        """
        result_dct = self.nested_dict()

        for root, dirs, files in os.walk(self.shap_ia_values_path):
            if str(self.shap_ia_values_file_name) in files:
                relative_path = os.path.relpath(root, self.shap_ia_values_path)
                parts = relative_path.split(os.sep)

                if len(parts) == 4:
                    feature_combination, samples, crit, model = parts

                    if (
                        crit == crit_to_plot
                        and samples == samples_to_include
                        and model == model_to_plot
                        and feature_combination == feature_combination_to_plot
                    ):
                        shap_ia_values_path = os.path.join(
                            str(root), str(self.shap_ia_values_file_name)
                        )
                        shap_ia_values = self.data_loader.read_pkl(shap_ia_values_path)

                        base_value = self.get_base_values(root)

                        shap_ia_values_dct = {
                            key: value["mean"]
                            for key, value in shap_ia_values["top_interactions"][
                                "top_abs_interactions_of_sample"
                            ].items()
                            if isinstance(key, tuple) and len(key) > 1
                        }

                        shap_ia_values_dct = dict(
                            sorted(
                                shap_ia_values_dct.items(),
                                key=lambda item: abs(np.mean(item[1])),
                                reverse=True,
                            )
                        )
                        feature_tuples = list(shap_ia_values_dct.keys())

                        formatted_features_dct = {}
                        for feature_pair, values in shap_ia_values_dct.items():
                            mapped_key = " x ".join(
                                apply_name_mapping([k], self.name_mapping, prefix=True)[
                                    0
                                ]
                                for k in feature_pair
                            )
                            formatted_features_dct[mapped_key] = values

                        shap_ia_values_arr = np.array(
                            [value for value in formatted_features_dct.values()]
                        )
                        feature_names = list(formatted_features_dct.keys())

                        data = self.get_ia_feature_data(
                            root_path=root, top_n_interactions=feature_tuples
                        )
                        shap_ia_exp = self.recreate_shap_exp_objects(
                            shap_values=shap_ia_values_arr.T,
                            base_values=np.array(base_value),
                            feature_names=feature_names,
                            data=data.values,
                        )

                        result_dct[f"{feature_combination}_ia_values"] = shap_ia_exp

        return defaultdict_to_dict(result_dct)

    def get_ia_feature_data(
        self, root_path: str, top_n_interactions: list[tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Loads feature values for SHAP interaction analysis and computes the mean of specified feature pairs for the beeswarm plot.

        Args:
            root_path: Path to the directory containing the SHAP values file.
            top_n_interactions: List of tuples representing the strongest feature interactions.

        Returns:
            pd.DataFrame: DataFrame containing new columns with the mean values of the specified feature pairs.

        Raises:
            ValueError: If an interaction does not contain exactly two features or if features are missing from the DataFrame.
        """
        file_name = os.path.join(root_path, self.shap_values_file_name)
        shap_values = self.data_loader.read_pkl(file_name)
        data = shap_values["data"]["mean"]
        feature_names = [
            feature
            for feature in shap_values["feature_names"]
            if feature not in self.meta_vars
        ]

        feature_df = pd.DataFrame(data, columns=feature_names)

        for interaction in top_n_interactions:
            if len(interaction) != 2:
                raise ValueError(
                    f"Each interaction must contain exactly two feature names. Invalid entry: {interaction}"
                )

            feature1, feature2 = interaction
            if feature1 not in feature_df.columns or feature2 not in feature_df.columns:
                raise ValueError(
                    f"Features {feature1} and {feature2} must be present in the DataFrame columns."
                )

            interaction_col_name = (feature1, feature2)
            feature_df[interaction_col_name] = feature_df[[feature1, feature2]].mean(
                axis=1
            )

        feature_df = feature_df.drop(columns=feature_names)

        return feature_df

    def get_base_values(self, root_path: str) -> list[float]:
        """
        Retrieves the base values for SHAP interaction analysis from a specified directory.

        Args:
            root_path: The path to the directory containing the SHAP values file.

        Returns:
            list[float]: A list of base values corresponding to the SHAP interaction analysis.
        """
        file_name = os.path.join(root_path, self.shap_values_file_name)
        shap_values = self.data_loader.read_pkl(file_name)

        return shap_values["base_values"]["mean"]

    @staticmethod
    def get_required_sample_for_combo(
        feature_combination: str, col_assignment: dict[str, list[str]]
    ) -> str:
        """
        Selects
        For the plot in the paper, we need to adjust "samples_to_include" based on the specific feature combination.
        Therefore, we need a custom mapping to load the right data

        Args:
            feature_combination: str, the current feature_combination
            col_assignment: list, the feature_combination and its associated location in the plot

        Returns:
            str: samples_to_include for the current feature_combination
        """
        if feature_combination in col_assignment["first_col"]:
            return "selected"

        for sublst in col_assignment["second_col"] + col_assignment["third_col"]:
            if feature_combination in sublst:
                return "all"

    @staticmethod
    def recreate_shap_exp_objects(
        shap_values: np.ndarray,
        base_values: np.ndarray,
        feature_names: list = None,
        data: np.ndarray = None,
    ) -> Explanation:
        """
        Recreates the SHAP explanation objects from the data.

        Args:
            shap_values: 2darray, containing the shap_values
            base_values: 1darray, containing the base_values
            data: 2darray, containing the data
            feature_names: list, containing the feature names corresponding to the shap values

        Returns:
            SHAP.Explanation object
        """
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_values,
            data=data,
            feature_names=feature_names,
        )
        return explanation
