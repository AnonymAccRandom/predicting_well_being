import os
from typing import Any

import numpy as np

from src.utils.DataLoader import DataLoader
from src.utils.DataSaver import DataSaver
from src.utils.utilfuncs import apply_name_mapping, NestedDict


class SuppFileCreator:
    """
    Handles the creation of supplementary files for data analysis by processing and formatting specific
    input files (e.g., linear model coefficients, SHAP values, SHAP interaction values) and mirroring the
    directory structure in the output location.

    Attributes:
        cfg_postprocessing (NestedDict): Configuration settings for postprocessing.
        name_mapping (NestedDict): Mapping of feature names to their formatted equivalents for presentation.
        data_loader (DataLoader): Instance of the DataLoader class for reading input files.
    """

    def __init__(
        self,
        cfg_postprocessing: NestedDict,
        name_mapping: NestedDict,
        meta_vars: list[str],
    ) -> None:
        """
        Initializes the SuppFileCreator class.

        Args:
            cfg_postprocessing: Configuration dictionary for postprocessing.
            name_mapping: Mapping of feature names for display purposes.
            meta_vars: List of meta_vars to exclude from feature names.
        """
        self.name_mapping = name_mapping
        self.cfg_postprocessing = cfg_postprocessing

        self.data_loader = DataLoader()
        self.data_saver = DataSaver()

        self.meta_vars = meta_vars

    def create_mirrored_dir_with_files(
        self,
        base_dir: str,
        file_name: str,
        output_base_dir: str,
    ) -> None:
        """
        Creates a mirrored directory structure at the output location and processes specific files.

        Args:
            base_dir: Base directory to traverse for input files.
            file_name: Name of the file to search for and process.
            output_base_dir: Base directory for saving processed files.

        Raises:
            ValueError: If the file type is not supported for processing.
        """
        for root, _, files in os.walk(base_dir):
            if file_name in files:
                relative_path = os.path.relpath(root, base_dir)
                target_dir = os.path.join(output_base_dir, relative_path)
                os.makedirs(target_dir, exist_ok=True)

                input_file_path = os.path.join(root, file_name)
                output_file_path = os.path.join(target_dir, file_name)

                if file_name.startswith("lin_model_coefs"):
                    self.process_lin_model_coefs_for_supp(
                        input_file_path, output_file_path
                    )

                elif file_name.startswith("shap_values"):
                    self.process_shap_values_for_supp(input_file_path, output_file_path)

                elif file_name.startswith("shap_ia_values"):
                    self.process_shap_ia_values_for_supp(
                        input_file_path, output_file_path
                    )

                else:
                    raise ValueError(f"Input file {file_name} not supported yet")

    def process_lin_model_coefs_for_supp(
        self, input_file_path: str, output_file_path: str
    ) -> None:
        """
        Processes lin_model_coefs by replacing feature names with formatted versions

        Args:
            input_file_path: Path to the input JSON file containing linear model coefficients.
            output_file_path: Path to save the processed JSON file.
        """
        lin_model_coefs = self.data_loader.read_json(input_file_path)

        for stat, vals in lin_model_coefs.items():
            new_feature_names = apply_name_mapping(
                features=list(vals.keys()), name_mapping=self.name_mapping, prefix=True
            )
            updated_vals = {
                new_name: vals[old_name]
                for old_name, new_name in zip(vals.keys(), new_feature_names)
            }

            lin_model_coefs[stat] = updated_vals

        self.data_saver.save_json(lin_model_coefs, output_file_path)

    def process_shap_values_for_supp(
        self, input_file_path: str, output_file_path: str
    ) -> None:
        """
        Processes SHAP values by replacing feature names and storing the formatted versions.

        Args:
            input_file_path: Path to the input pickle file containing SHAP values.
            output_file_path: Path to save the processed pickle file.
        """
        shap_values = self.data_loader.read_pkl(input_file_path)
        feature_names_copy = shap_values["feature_names"].copy()
        feature_names_copy = [
            feature for feature in feature_names_copy if feature not in self.meta_vars
        ]

        # Convert lists to arras for more efficient storage
        for key, values in shap_values.items():
            if key == "feature_names":
                continue
            for stat, stat_vals in values.items():
                shap_values[key][stat] = np.array(stat_vals, dtype=np.float16)

        formatted_feature_names = apply_name_mapping(
            features=feature_names_copy,
            name_mapping=self.name_mapping,
            prefix=True,
        )
        shap_values["feature_names"] = formatted_feature_names

        self.data_saver.save_pickle(shap_values, output_file_path)

    def process_shap_ia_values_for_supp(
        self, input_file_path: str, output_file_path: str
    ) -> None:
        """
        Processes SHAP interaction values by renaming keys that correspond to specific features.

        Args:
            input_file_path: Path to the input pickle file containing SHAP interaction values.
            output_file_path: Path to save the processed pickle file.
        """
        shap_ia_values = self.data_loader.read_pkl(input_file_path)
        srmc_name_mapping = {
            f"srmc_{feature}": feature_formatted
            for feature, feature_formatted in self.name_mapping["srmc"].items()
        }
        renamed_ia_values = self.rename_srmc_keys(shap_ia_values, srmc_name_mapping)

        self.data_saver.save_pickle(renamed_ia_values, output_file_path)

    def rename_srmc_keys(self, data: Any, srmc_name_mapping: dict[str, str]) -> Any:
        """
        Recursively traverses a nested dictionary and renames keys based on a mapping for 'srmc' prefixed strings.

        This method processes the keys in a nested dictionary structure and applies renaming rules:
        - If a key is a string that starts with 'srmc', it is replaced using the provided `srmc_name_mapping`.
        - If a key is a tuple, any string element within the tuple that starts with 'srmc' is replaced using the mapping.
        - Other keys (e.g., non-'srmc' strings, integers) are left unchanged.

        Args:
            data: A nested dictionary (or other data structure) containing the keys to be renamed.
            srmc_name_mapping: A mapping of 'srmc' prefixed keys to their replacement names.

        Returns:
            Any: A new dictionary (or the original data structure) with renamed keys where applicable.
        """
        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                if isinstance(key, str) and key.startswith("srmc"):
                    new_key = srmc_name_mapping.get(key, key)

                elif isinstance(key, tuple):
                    replaced_tuple = []
                    for part in key:
                        if isinstance(part, str) and part.startswith("srmc"):
                            replaced_tuple.append(srmc_name_mapping.get(part, part))
                        else:
                            replaced_tuple.append(part)
                    new_key = tuple(replaced_tuple)

                else:
                    new_key = key
                new_dict[new_key] = self.rename_srmc_keys(value, srmc_name_mapping)

            return new_dict

        # If `data` is not a dict, just return it as-is (e.g., int, str, list, etc.)
        return data
