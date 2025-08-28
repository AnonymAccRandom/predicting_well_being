import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from src.utils.utilfuncs import NestedDict


class ClusterSummarizer:
    """
    A class for summarizing results generated on a computational cluster.

    This class aggregates results from multiple repetitions and imputations for cross-validation (CV) metrics,
    linear model coefficients, SHAP values, and SHAP interaction (IA) values.
    Note: We apply this function directly on the cluster, as we do not store all raw results locally.

    Key functionalities:
    - Computes mean (M) and standard deviation (SD) of CV results.
    - Aggregates coefficients from linear models across repetitions.
    - Aggregates SHAP values and SHAP IA values for interpretability analysis.

    Attributes:
        base_result_dir (str): The base directory where results are stored.
        cfg_analysis (NestedDict): Configuration dictionary loaded from the variable configuration YAML file.
        num_reps (int): Number of repetitions expected for the results.
        rng_ (np.random.RandomState): Random number generator seeded for reproducibility.
    """

    def __init__(
        self,
        base_result_dir: str,
        cfg_analysis_path: str = "../../configs/cfg_analysis.yaml",
        num_reps: int = 10,
    ) -> None:
        """
        Initializes the ClusterSummarizer with the base directory and configuration.

        Args:
            base_result_dir: Directory containing the result files to be aggregated.
            cfg_analysis_path: Path to the variable configuration YAML file.
            num_reps: Number of repetitions expected for the results.
        """
        self.base_result_dir = base_result_dir
        with open(cfg_analysis_path, "r") as f:
            cfg_analysis = yaml.safe_load(f)

        self.cfg_analysis = cfg_analysis
        self.num_reps = num_reps
        self.rng_ = np.random.RandomState(self.cfg_analysis["random_state"])

    def aggregate(self) -> None:
        """
        Walks through the directory tree starting from `self.base_result_dir` and aggregates results.

        The method checks for specific file types (e.g., CV results, linear model coefficients, SHAP values),
        processes them if files exist for all repetitions, and saves aggregated summaries.
        It prints intermediate steps so we see them directly in the consolde when executing the script
        on a supercomputer cluster.

        Steps:
        - Summarizes CV results if `cv_results_rep_{i}.json` files exist for all repetitions.
        - Aggregates linear model coefficients if `best_models_rep_{i}.pkl` files exist for all repetitions.
        - Aggregates SHAP values if `shap_values_rep_{i}.pkl` files exist for all repetitions.
        - Aggregates SHAP IA values if `shap_ia_values_rep_{i}.pkl` and corresponding base values exist.

        Results are stored in the respective directories as JSON or pickle files.
        """
        for root, dirs, files in os.walk(self.base_result_dir):
            print("---")
            print(root, dirs)

            # Summarize CV results
            if all(f"cv_results_rep_{i}.json" in files for i in range(self.num_reps)):
                cv_results_files = [
                    f"cv_results_rep_{i}.json" for i in range(self.num_reps)
                ]
                print(f"process cv_results in {root}")
                summary = self.summarize_cv_results(root, cv_results_files)

                with open(os.path.join(root, "cv_results_summary.json"), "w") as f:
                    json.dump(summary, f, indent=4)
                print(f"stored summarized cv_results in {root}")

            # Summarize linear model coefficients
            if all(f"best_models_rep_{i}.pkl" in files for i in range(self.num_reps)):
                if "elasticnet" in root:
                    best_models_files = [
                        f"best_models_rep_{i}.pkl" for i in range(self.num_reps)
                    ]
                    print(f"process lin_model_coefs in {root}")
                    summary = self.summarize_lin_model_coefs(root, best_models_files)

                    with open(
                        os.path.join(root, "lin_model_coefs_summary.json"), "w"
                    ) as f:
                        json.dump(summary, f, indent=4)
                    print(f"stored summarized lin_model_coefs in {root}")

            # Summarize SHAP values
            if all(f"shap_values_rep_{i}.pkl" in files for i in range(self.num_reps)):
                shap_files = [f"shap_values_rep_{i}.pkl" for i in range(self.num_reps)]
                print(f"process SHAP values in {root}")
                summary = self.summarize_shap_values(root, shap_files)

                with open(os.path.join(root, "shap_values_summary.pkl"), "wb") as f:
                    pickle.dump(summary, f)
                print(f"stored summarized shap_values in {root}")

            # Summarize SHAP IA Values
            if all(
                f"shap_ia_values_rep_{i}.pkl" in files for i in range(self.num_reps)
            ):
                shap_ia_values_files = [
                    f"shap_ia_values_rep_{i}.pkl" for i in range(self.num_reps)
                ]
                shap_ia_base_values_files = [
                    f"shap_ia_base_values_rep_{i}.pkl" for i in range(self.num_reps)
                ]
                print(f"process SHAP IA values in {root}")
                summary = self.summarize_shap_ia_values(
                    root, shap_ia_values_files, shap_ia_base_values_files
                )

                with open(os.path.join(root, "shap_ia_values_summary.pkl"), "wb") as f:
                    pickle.dump(summary, f)
                print(f"stored summarized shap_ia_values in {root}")

    def summarize_metrics(
        self, data_records: list[dict[str, int]], identifier_cols: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Computes summary statistics (mean and standard deviation) from a list of metric records.

        Args:
            data_records: List of dictionaries containing metrics and identifiers (e.g., repetitions, folds).
            identifier_cols: List of columns that act as identifiers (e.g., 'rep', 'outer_fold').

        Returns:
            dict: A dictionary containing:
                - Mean of all metrics.
                - Standard deviation across folds and imputations.
                - Standard deviation within folds across imputations.
                - Standard deviation across folds within imputations.
        """
        df = pd.DataFrame(data_records)

        metric_cols = [col for col in df.columns if col not in identifier_cols]

        overall_mean = df[metric_cols].mean().round(4).to_dict()
        overall_std = df[metric_cols].std(ddof=0).round(5).to_dict()

        sd_within_folds = df.groupby(["rep", "outer_fold"])[metric_cols].std(ddof=0)
        sd_within_folds_across_imps = sd_within_folds.mean().round(5).to_dict()

        sd_across_folds = df.groupby(["rep", "imputation"])[metric_cols].std(ddof=0)
        sd_across_folds_within_imps = sd_across_folds.mean().round(5).to_dict()

        summary = {
            "m": overall_mean,
            "sd_across_folds_imps": overall_std,
            "sd_within_folds_across_imps": sd_within_folds_across_imps,
            "sd_across_folds_within_imps": sd_across_folds_within_imps,
        }

        return summary

    def summarize_cv_results(
        self, result_dir: str, file_names: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Aggregates and summarizes cross-validation (CV) results from multiple files.

        Args:
            result_dir: Directory containing the result files.
            file_names: Names of the CV result files.

        Returns:
            dict: A dictionary with the following keys (for all 1st level key-value pairs, the value dict includes all metrics)
                - 'm': Mean CV results
                - 'sd_across_folds_imps': Standard deviation across folds and imputations.
                - 'sd_within_folds_across_imps': Standard deviation within folds across imputations.
                - 'sd_across_folds_within_imps': Standard deviation across folds within imputations.
        """
        data_records = []

        for rep_idx, file_name in enumerate(file_names):
            file_path = os.path.join(result_dir, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)

            for outer_fold_key, imps in data.items():
                outer_fold_idx = int(outer_fold_key.split("_")[-1])

                for imp_key, metrics in imps.items():
                    imp_idx = int(imp_key.split("_")[-1])
                    record = {
                        "rep": rep_idx,
                        "outer_fold": outer_fold_idx,
                        "imputation": imp_idx,
                    }
                    record.update(metrics)
                    data_records.append(record)

        identifier_cols = ["rep", "outer_fold", "imputation"]
        summary = self.summarize_metrics(data_records, identifier_cols)

        return summary

    def summarize_lin_model_coefs(
        self, result_dir: str, file_names: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Aggregates coefficients from linear models across repetitions, folds, and imputations.

        This method:
        - Loads linear model files containing coefficients for different outer folds, imputations, and repetitions.
        - Aggregates coefficients across these hierarchical levels.
        - Computes summary statistics including:
            - Mean and standard deviation for each coefficient across folds and imputations.
            - Count of non-zero coefficients for each feature.
        - Sorts features based on the absolute value of their mean coefficients for easier interpretability.

        Args:
            result_dir: Directory containing the coefficient files.
            file_names: Names of the coefficient files (one for each repetition).

        Returns:
            dict: A dictionary with the following keys:
                - 'm': Mean coefficients sorted by absolute value.
                - 'sd_across_folds_imps': Standard deviation across folds and imputations.
                - 'sd_within_folds_across_imps': Standard deviation within folds across imputations.
                - 'sd_across_folds_within_imps': Standard deviation across folds within imputations.
                - 'non_zero_coefs': Count of non-zero coefficients for each feature.
        """
        data_records = []

        for rep_idx, file_name in enumerate(file_names):
            file_path = os.path.join(result_dir, file_name)
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            for outer_fold_idx, imps in enumerate(data):
                for imp_idx, model in enumerate(imps):
                    if hasattr(model, "coef_"):
                        coefs = dict(zip(model.feature_names_in_, model.coef_.ravel()))
                        record = {
                            "rep": rep_idx,
                            "outer_fold": outer_fold_idx,
                            "imputation": imp_idx,
                        }

                        for feature_name, coef_value in coefs.items():
                            record[feature_name] = coef_value
                        data_records.append(record)

        identifier_cols = ["rep", "outer_fold", "imputation"]
        summary = self.summarize_metrics(data_records, identifier_cols)
        df = pd.DataFrame(data_records)

        metric_cols = [col for col in df.columns if col not in identifier_cols]
        non_zero_counts = (df[metric_cols] != 0).sum().to_dict()

        sorted_features = sorted(
            summary["m"].keys(), key=lambda k: abs(summary["m"][k]), reverse=True
        )

        summary["m"] = {k: summary["m"][k] for k in sorted_features}
        summary["sd_across_folds_imps"] = {
            k: summary["sd_across_folds_imps"][k] for k in sorted_features
        }
        summary["sd_within_folds_across_imps"] = {
            k: summary["sd_within_folds_across_imps"][k] for k in sorted_features
        }
        summary["sd_across_folds_within_imps"] = {
            k: summary["sd_across_folds_within_imps"][k] for k in sorted_features
        }
        summary["non_zero_coefs"] = {
            k: non_zero_counts.get(k, 0) for k in sorted_features
        }

        return summary

    @staticmethod
    def summarize_shap_values(result_dir: str, file_names: list[str]) -> NestedDict:
        """
        Aggregates SHAP values across repetitions and imputations.

        This method:
        - Reads SHAP data files containing SHAP values, base values, and the corresponding dataset.
        - Aggregates SHAP values across repetitions and imputations.
        - Computes the mean and standard deviation for SHAP values and base values across these dimensions.
        - Ensures that the aggregated results maintain their structure for interpretability.

        Args:
            result_dir: Directory containing the SHAP value files.
            file_names: Names of the SHAP value files (one for each repetition).

        Returns:
            dict: A dictionary with the following keys:
                - 'shap_values': Mean and standard deviation of SHAP values.
                - 'data': Mean and standard deviation of the dataset values.
                - 'base_values': Mean and standard deviation of the base values.
                - 'feature_names': List of feature names corresponding to the SHAP values.
        """
        shap_values_list = []
        data_list = []
        base_values_list = []
        feature_names = None

        for file_name in file_names:
            file_path = os.path.join(result_dir, file_name)
            with open(file_path, "rb") as f:
                shap_data = pickle.load(f)

            if "shap_values" in shap_data:
                shap_values_list.append(
                    np.expand_dims(shap_data["shap_values"], axis=0)
                )

            if "data" in shap_data:
                data_list.append(np.expand_dims(shap_data["data"], axis=0))

            if "base_values" in shap_data:
                base_values_list.append(
                    np.expand_dims(shap_data["base_values"], axis=0)
                )

            if feature_names is None:
                feature_names = shap_data.get("feature_names")

        shap_values_array = np.concatenate(
            shap_values_list, axis=0
        )  # Shape: (reps, ...)
        data_array = np.concatenate(data_list, axis=0)  # Shape: (reps, ...)
        base_values_array = np.concatenate(
            base_values_list, axis=0
        )  # Shape: (reps, ...)

        results = {}
        for key, values in zip(
            ["shap_values", "data", "base_values"],
            [shap_values_array, data_array, base_values_array],
        ):
            imp_axis = 3 if values.ndim > 3 else 2

            results[key] = {
                "mean": np.mean(
                    values, axis=(0, imp_axis)
                ).tolist(),  # Average across repetitions and imputations
                "std": np.std(
                    values, axis=(0, imp_axis)
                ).tolist(),  # Std dev across repetitions and imputations
            }

        results["feature_names"] = feature_names

        return results

    def summarize_shap_ia_values(
        self,
        result_dir: str,
        shap_ia_values_files: list[str],
        shap_ia_base_values_files: list[str],
    ) -> NestedDict:
        """
        Aggregates SHAP interaction (IA) values across repetitions and imputations.

        This method:
        - Loads SHAP IA value files and their corresponding base values for all repetitions.
        - Aggregates these values across repetitions and imputations.
        - Computes various metrics, including absolute and raw interaction values for different feature combinations.
        - Identifies the most interacting features and the most important interactions.

        Args:
            result_dir: Directory containing the SHAP IA value files.
            shap_ia_values_files: List of SHAP IA value files (one per repetition).
            shap_ia_base_values_files: List of SHAP IA base value files (one per repetition).

        Returns:
            dict: A dictionary containing processed SHAP IA results with keys:
                - 'ia_values_sample': Randomly sampled aggregated IA values.
                - 'base_values_sample': Corresponding base values for the sampled IA values.
                - 'top_interactions': The top feature interactions ranked by their importance.
                - 'top_interacting_features': Features involved in the most significant interactions.
                - 'abs_ia_value_agg_reps_imps_samples': Aggregated absolute IA values across repetitions and imputations.
        """
        ia_values_all_reps = {}
        base_values_all_reps = {}

        mapping_path = os.path.join(result_dir, "ia_values_mappings_rep_0.pkl")
        with open(mapping_path, "rb") as file:
            mapping_dct = pickle.load(file)
            combo_index_mapping = mapping_dct["combo_index_mapping"]
            feature_index_mapping = mapping_dct["feature_index_mapping"]

        for rep_idx, file_name in enumerate(shap_ia_values_files):
            file_path = os.path.join(result_dir, file_name)

            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    shap_ia_results_reps_imps = pickle.load(file)
                ia_values_all_reps[f"rep_{rep_idx}"] = shap_ia_results_reps_imps

            else:
                print(
                    f"Warning: SHAP IA values file for repetition {rep_idx} not found."
                )
                continue

        for rep_idx, file_name in enumerate(shap_ia_base_values_files):
            file_path = os.path.join(result_dir, file_name)

            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    base_values_reps_imps = pickle.load(file)
                base_values_all_reps[f"rep_{rep_idx}"] = base_values_reps_imps

            else:
                print(
                    f"Warning: SHAP IA base values file for repetition {rep_idx} not found."
                )
                continue

        # Aggregate across repetitions and imputations
        (
            ia_values_agg_reps_imps,
            base_value_agg_reps_imps,
        ) = self.agg_ia_values_across_reps(
            ia_value_dct=ia_values_all_reps,
            base_value_dct=base_values_all_reps,
        )

        # Aggregate results across samples
        (
            abs_ia_values_agg_reps_imps_samples,
            ia_values_agg_reps_imps_samples,
            abs_base_value_agg_reps_imps_samples,
            base_value_agg_reps_imps_samples,
        ) = self.agg_results_across_samples(
            ia_value_dct=ia_values_agg_reps_imps.copy(),
            base_value_dct=base_value_agg_reps_imps.copy(),
        )

        # Map combinations to features - abs ia values across samples
        abs_ia_values_agg_reps_imps_samples = self.map_combo_to_features(
            abs_ia_values_agg_reps_imps_samples,
            feature_index_mapping=feature_index_mapping,
            combo_index_mapping=combo_index_mapping,
        )
        # Map combinations to features - raw ia values across samples
        ia_values_agg_reps_imps_samples = self.map_combo_to_features(
            ia_values_agg_reps_imps_samples,
            feature_index_mapping=feature_index_mapping,
            combo_index_mapping=combo_index_mapping,
        )
        # Map combinations to features - ia values not aggregated across samples
        ia_values_agg_reps_imps = self.map_combo_to_features(
            ia_values_agg_reps_imps,
            feature_index_mapping=feature_index_mapping,
            combo_index_mapping=combo_index_mapping,
        )
        # Sample aggregated results
        num_samples = self.cfg_analysis["shap_ia_values"]["num_samples"]
        ia_values_sample, base_values_sample = self.sample_aggregated_results(
            ia_values_agg_reps_imps, base_value_agg_reps_imps, num_samples=num_samples
        )

        # Prepare results dictionary to return
        shap_ia_results_processed = {
            "ia_values_sample": ia_values_sample,
            "base_values_sample": base_values_sample,
        }

        # Get top interactions # here we want the abs and the raw means
        top_interactions_per_order = self.get_top_interactions(
            abs_mapped_results_agg_samples=abs_ia_values_agg_reps_imps_samples,
            mapped_results_agg_samples=ia_values_agg_reps_imps_samples,
            mapped_results_no_agg_samples=ia_values_agg_reps_imps,
        )

        # Get most interacting features # here we only need the abs means
        top_interacting_features = self.get_most_interacting_features(
            abs_ia_values_agg_reps_imps_samples
        )

        shap_ia_results_processed["top_interactions"] = top_interactions_per_order
        shap_ia_results_processed["top_interacting_features"] = top_interacting_features
        shap_ia_results_processed["abs_ia_value_agg_reps_imps_samples"] = (
            abs_ia_values_agg_reps_imps_samples
        )

        print("Processed all SHAP IA values.")

        return shap_ia_results_processed

    @staticmethod
    def agg_ia_values_across_reps(
        ia_value_dct: dict[str, np.ndarray], base_value_dct: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Aggregates interaction attribution (IA) values and base values across repetitions.

        This method:
        - Takes SHAP interaction and base value dictionaries, where each repetition is a key.
        - Aggregates the values across repetitions to calculate the mean and standard deviation.
        - Preserves the sample and feature combination structure for downstream analyses.

        Args:
            ia_value_dct: Dictionary with keys as repetitions (e.g., "rep_0") and values as ndarrays of shape
                                 (samples, combinations, imputations).
            base_value_dct: Dictionary with keys as repetitions (e.g., "rep_0") and values as ndarrays of shape
                                   (samples, imputations).

        Returns:
            tuple[dict, dict]: Two dictionaries:
                - Aggregated IA values with keys 'mean' and 'std'.
                - Aggregated base values with keys 'mean' and 'std'.
        """
        ia_values_array = np.stack(
            list(ia_value_dct.values()), axis=-1
        )  # Shape: (samples, combinations, imputations, repetitions)

        ia_mean = np.mean(
            ia_values_array, axis=-1
        )  # Resulting shape: (samples, combinations)
        ia_std = np.std(
            ia_values_array, axis=-1
        )  # Resulting shape: (samples, combinations)

        base_values_array = np.stack(
            list(base_value_dct.values()), axis=-1
        )  # Shape: (samples, imputations, repetitions)

        base_mean = np.mean(base_values_array, axis=-1)  # Resulting shape: (samples,)
        base_std = np.std(base_values_array, axis=-1)  # Resulting shape: (samples,)

        return ({"mean": ia_mean, "std": ia_std}, {"mean": base_mean, "std": base_std})

    @staticmethod
    def agg_results_across_samples(
        ia_value_dct: dict[str, np.ndarray], base_value_dct: dict[str, np.ndarray]
    ) -> tuple[dict, dict, dict, dict]:
        """
        Aggregates IA and base values across samples to compute mean and standard deviation.

        This method:
        - Calculates the absolute and raw mean IA values across samples for each feature combination.
        - Computes standard deviations to capture variability across samples.
        - Performs similar operations for base values.

        Args:
            ia_value_dct: Aggregated IA values with keys 'mean' and 'std', where 'mean' and 'std' are arrays
                          shaped as (samples, combinations).
            base_value_dct: Aggregated base values with keys 'mean' and 'std', where 'mean' and 'std' are arrays
                            shaped as (samples,).

        Returns:
            tuple[dict, dict, dict, dict]: Four dictionaries:
                - Absolute mean and std of IA values across samples.
                - Raw mean and std of IA values across samples.
                - Absolute mean and std of base values across samples.
                - Raw mean and std of base values across samples.
        """
        ia_mean = ia_value_dct["mean"]  # Shape: (samples, combinations)

        ia_abs_mean_across_samples = {
            "mean": np.mean(
                np.abs(ia_mean), axis=0
            ),  # Mean of absolute values, shape: (combinations,)
            "std": np.std(
                np.abs(ia_mean), axis=0
            ),  # Std of absolute values, shape: (combinations,)
        }

        ia_raw_mean_across_samples = {
            "mean": np.mean(
                ia_mean, axis=0
            ),  # Mean of raw values, shape: (combinations,)
            "std": np.std(ia_mean, axis=0),  # Std of raw values, shape: (combinations,)
        }

        base_mean = base_value_dct["mean"]  # Shape: (samples,)

        base_abs_mean_across_samples = {
            "mean": np.mean(np.abs(base_mean)),  # Scalar
            "std": np.std(np.abs(base_mean)),  # Scalar
        }

        base_raw_mean_across_samples = {
            "mean": np.mean(base_mean),  # Scalar
            "std": np.std(base_mean),  # Scalar
        }

        return (
            ia_abs_mean_across_samples,
            ia_raw_mean_across_samples,
            base_abs_mean_across_samples,
            base_raw_mean_across_samples,
        )

    def sample_aggregated_results(
        self, ia_value_dct: NestedDict, base_value_dct: NestedDict, num_samples: int
    ) -> tuple[NestedDict, NestedDict]:
        """
        Samples a fixed number of IA and base values from the aggregated results.

        This method:
        - Randomly selects a specified number of samples from the aggregated IA and base values.
        - Ensures the sampled values preserve their original structure (e.g., mean and std for each feature combination).

        Args:
            ia_value_dct: Aggregated IA values with 'mean' and 'std' for each combination.
            base_value_dct: Aggregated base values with 'mean' and 'std'.
            num_samples: The number of samples to randomly select.

        Returns:
            tuple[NestedDict, NestedDict]: Two dictionaries:
                - Sampled IA values for each combination.
                - Sampled base values.
        """
        total_samples = next(iter(ia_value_dct.values()))["mean"].shape[0]
        sampled_indices = self.rng_.choice(
            total_samples, size=num_samples, replace=False
        )

        sampled_ia_values = {}
        sampled_base_values = {}

        for combination, stats in ia_value_dct.items():
            ia_mean_sampled = stats["mean"][sampled_indices]
            ia_std_sampled = stats["std"][sampled_indices]

            sampled_ia_values[combination] = {
                "mean": ia_mean_sampled,
                "std": ia_std_sampled,
            }

        sampled_base_values["mean"] = base_value_dct["mean"][sampled_indices]
        sampled_base_values["std"] = base_value_dct["std"][sampled_indices]

        return sampled_ia_values, sampled_base_values

    def map_combo_to_features(
        self,
        ia_mean_across_samples: dict[str, np.ndarray],
        feature_index_mapping: dict[int, str],
        combo_index_mapping: dict[int, tuple[int, int]],
    ) -> NestedDict:
        """
        Maps feature combinations (index-based) to their corresponding feature names.

        This method:
        - Converts numeric indices of feature combinations into readable feature names.
        - Supports both aggregated (1D array) and non-aggregated (2D array) IA values.

        Args:
            ia_mean_across_samples: Dictionary with 'mean' and 'std' arrays for IA values, which can be:
                - 1D: Aggregated across samples.
                - 2D: Contains values for individual samples.
            feature_index_mapping: Maps feature indices to feature names.
            combo_index_mapping: Maps combination indices to tuples of feature indices.

        Returns:
            dict: A dictionary where keys are tuples of feature names, and values are dicts containing 'mean' and 'std' arrays.
        """
        result = {}
        mean_array = ia_mean_across_samples["mean"]
        std_array = ia_mean_across_samples["std"]

        match mean_array.ndim:
            case 1:
                num_combinations = len(mean_array)
                for combination_index in range(num_combinations):
                    feature_indices = combo_index_mapping[combination_index]
                    feature_names = tuple(
                        feature_index_mapping[idx] for idx in feature_indices
                    )

                    mean_value = mean_array[combination_index]
                    std_value = std_array[combination_index]

                    result[feature_names] = {"mean": mean_value, "std": std_value}

            case 2:
                num_samples, num_combinations = mean_array.shape
                for combination_index in range(num_combinations):
                    feature_indices = combo_index_mapping[combination_index]
                    feature_names = tuple(
                        feature_index_mapping[idx] for idx in feature_indices
                    )

                    mean_values = mean_array[:, combination_index]
                    std_values = std_array[:, combination_index]

                    result[feature_names] = {"mean": mean_values, "std": std_values}

            case _:
                raise ValueError(
                    "Expected mean array to be 1D or 2D, but got an array with shape: "
                    f"{mean_array.shape}"
                )
        return result

    @staticmethod
    def get_top_interactions(
        abs_mapped_results_agg_samples: NestedDict,
        mapped_results_agg_samples: NestedDict,
        mapped_results_no_agg_samples: NestedDict,
        top_n: int = 20,
    ) -> NestedDict:
        """
        Identifies the top N most important feature interactions.

        This method:
        - Extracts interactions with the highest absolute mean IA values for each interaction order (e.g., pairwise, triplet).
        - Separately identifies the top N interactions based on raw values.
        - Filters non-aggregated results to include only the identified top interactions.

        Args:
            abs_mapped_results_agg_samples: Aggregated absolute IA values.
            mapped_results_agg_samples: Aggregated raw IA values.
            mapped_results_no_agg_samples: Non-aggregated IA values.
            top_n: Number of top interactions to extract for each interaction order.

        Returns:
            dict: A dictionary containing:
                - 'top_abs_interactions': Top absolute interactions by order.
                - 'top_raw_interactions': Top raw interactions by order.
                - 'top_abs_interactions_of_sample': Non-aggregated values for top absolute interactions.
                - 'top_raw_interactions_of_sample': Non-aggregated values for top raw interactions.
        """
        # Step 1: Group the interactions by their order (length of the feature tuple) for absolute values
        abs_order_dict = defaultdict(list)
        for feature_tuple, value_dict in abs_mapped_results_agg_samples.items():
            order = len(feature_tuple)
            mean_value = value_dict["mean"]
            std_value = value_dict["std"]
            abs_order_dict[order].append((feature_tuple, mean_value, std_value))

        # Step 2: Group the interactions by their order for raw values
        raw_order_dict = defaultdict(list)
        for feature_tuple, value_dict in mapped_results_agg_samples.items():
            order = len(feature_tuple)
            mean_value = value_dict["mean"]
            std_value = value_dict["std"]
            raw_order_dict[order].append((feature_tuple, mean_value, std_value))

        # Step 3: Extract the top N interactions for absolute values
        abs_res = {}
        abs_selected_interactions = set()
        for order, interactions in abs_order_dict.items():
            interactions_sorted = sorted(
                interactions, key=lambda x: abs(x[1]), reverse=True
            )

            top_interactions = interactions_sorted[:top_n]
            abs_selected_interactions.update(
                [feat_tuple for feat_tuple, _, _ in top_interactions]
            )

            abs_res[f"order_{order}"] = {
                feat_tuple: {"mean": mean_val, "std": std_val}
                for feat_tuple, mean_val, std_val in top_interactions
            }

        # Step 4: Extract the top N interactions for raw values
        raw_res = {}
        raw_selected_interactions = set()
        for order, interactions in raw_order_dict.items():
            interactions_sorted = sorted(
                interactions, key=lambda x: abs(x[1]), reverse=True
            )

            top_interactions = interactions_sorted[:top_n]

            raw_selected_interactions.update(
                [feat_tuple for feat_tuple, _, _ in top_interactions]
            )

            raw_res[f"order_{order}"] = {
                feat_tuple: {"mean": mean_val, "std": std_val}
                for feat_tuple, mean_val, std_val in top_interactions
            }

        # Step 5: Filter mapped_results_no_agg_samples for absolute and raw interactions
        abs_filtered_results = {
            feat_tuple: value_dict
            for feat_tuple, value_dict in mapped_results_no_agg_samples.items()
            if feat_tuple in abs_selected_interactions
        }

        raw_filtered_results = {
            feat_tuple: value_dict
            for feat_tuple, value_dict in mapped_results_no_agg_samples.items()
            if feat_tuple in raw_selected_interactions
        }

        # Step 6: Combine the results into the output dictionary
        final_res = {
            "top_abs_interactions": abs_res,
            "top_raw_interactions": raw_res,
            "top_abs_interactions_of_sample": abs_filtered_results,
            "top_raw_interactions_of_sample": raw_filtered_results,
        }

        return final_res

    def get_most_interacting_features(
        self, abs_mapped_results_agg_samples: NestedDict
    ) -> dict[str, float]:
        """
        Computes the most interacting features across all higher-order interactions (order > 1).

        This method:
        - Focuses on feature interactions of order greater than 1 (e.g., pairwise, triplets).
        - For each feature, aggregates the absolute mean IA values of all combinations where the feature appears.
        - Summarizes the importance of each feature based on its involvement in higher-order interactions.

        Implementation:
        - Iterates over all feature combinations (tuples) in `abs_mapped_results_agg_samples`.
        - Filters combinations to only include those with an interaction order greater than 1.
        - For each combination, adds the absolute mean IA value to the running total for each feature in the combination.
        - At the end, sorts the features by their total interaction importance in descending order.

        Args:
            abs_mapped_results_agg_samples: A dictionary mapping feature combinations (tuples) to their
                                            aggregated absolute IA values. Each value is a dictionary with:
                                               - 'mean': The aggregated mean IA value for the combination.
                                               - 'std': The aggregated standard deviation (not used here).

        Returns:
            dict[str, float]: A dictionary mapping individual feature names to their total summed absolute IA values,
                              sorted in descending order of interaction importance.
        """
        feature_interaction_sums = {}

        for feature_tuple, value_dict in abs_mapped_results_agg_samples.items():
            if len(feature_tuple) > 1:
                mean_value = value_dict["mean"]
                abs_mean_value = abs(mean_value)

                for feature in feature_tuple:
                    feature_interaction_sums[feature] = (
                        feature_interaction_sums.get(feature, 0.0) + abs_mean_value
                    )

        sorted_features = dict(
            sorted(
                feature_interaction_sums.items(), key=lambda item: item[1], reverse=True
            )
        )

        return sorted_features


if __name__ == "__main__":
    print("Hello")
    # adjust the path and num_reps as needed here
    summarizer = ClusterSummarizer(
        base_result_dir="../../results/local_tests/srmc/all/wb_state/", num_reps=10
    )
    summarizer.aggregate()
