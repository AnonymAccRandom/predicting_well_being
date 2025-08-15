import copy
import os
import re
from collections import defaultdict
from math import sqrt
from statistics import stdev
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import fdrcorrection

from src.utils.DataLoader import DataLoader
from src.utils.DataSaver import DataSaver
from src.utils.utilfuncs import (
    defaultdict_to_dict,
    NestedDict,
    create_defaultdict,
    format_p_values,
)


class SignificanceTesting:
    """
    Computes tests of significance to compare prediction results for different models
    across various analysis settings.

    This class performs the following:
    - Computes pairwise comparisons of models (corrected_paired_t_test), pooled across feature selection strategies.
    - Applies False Discovery Rate (FDR) correction to account for multiple testing.
    - Stores results in JSON format for annotations in CV result plots and as a table for supplementary results.

    Attributes:
        cfg_postprocessing (NestedDict): Configuration settings for postprocessing (parsed from a YAML file).
        cfg_sig (NestedDict): Configuration for conducting significance tests, extracted from `cfg_postprocessing`.
        sig_result_dir (str): Directory where the significance test results are stored.
        data_loader (DataLoader): Utility for loading data from files.
        data_saver (DataSaver): Utility for saving results in various formats.
        cv_file_matching_pattern (re.Pattern): Compiled regex pattern to match CV result file names.
        model_name_mapping (dict): Mapping of model names (e.g., `elasticnet` -> `ENR`).
        feature_combo_name_mapping_main (dict): Feature combination mappings for main analysis.
        crit (str): Criterion used for the significance tests (e.g., `wb_state`).
        metric (str): Metric used for comparisons (e.g., `r2`).
        models (list[str]): List of models to compare (e.g., `elasticnet`, `randomforestregressor`).
        samples_to_include (list[str]): List of sample subsets (e.g., `selected`, `control`).
        decimals (int): Number of decimal places for rounding results.
        delta_r2_strng (str): String representation for the delta R² value.
        t_strng (str): String representation for the t-value.
        p_strng (str): String representation for the p-value.
        p_fdr_strng (str): String representation for the FDR-corrected p-value.
    """

    def __init__(
        self,
        base_result_path: str,
        cfg_postprocessing: NestedDict,
    ) -> None:
        """
        Initializes the SignificanceTesting class.

        Args:
            base_result_path: Base path containing all results
            cfg_postprocessing: Configuration dictionary for postprocessing.

        Attributes Initialized:
            - Extracts key configuration values from `cfg_postprocessing`.
            - Compiles the regex pattern for matching CV result files.
            - Sets default mappings for models, feature combinations, and sample subsets.
        """
        self.cfg_postprocessing = cfg_postprocessing
        self.cfg_sig = self.cfg_postprocessing["conduct_significance_tests"]

        self.sig_result_dir = os.path.join(
            base_result_path,
            self.cfg_postprocessing["general"]["data_paths"]["sig_tests"],
        )

        self.data_loader = DataLoader()
        self.data_saver = DataSaver()

        self.cv_file_matching_pattern = re.compile(
            self.cfg_sig["cv_results_matching_pattern"]
        )

        self.model_name_mapping = self.cfg_postprocessing["general"]["models"][
            "name_mapping"
        ]
        self.feature_combo_name_mapping_main = self.cfg_postprocessing["general"][
            "feature_combinations"
        ]["name_mapping"]["main"]

        self.crit = self.cfg_sig["crit"]
        self.metric = self.cfg_sig["metric"]
        self.models = list(
            self.cfg_postprocessing["general"]["models"]["name_mapping"].keys()
        )
        self.samples_to_include = list(
            self.cfg_postprocessing["general"]["samples_to_include"][
                "name_mapping"
            ].keys()
        )
        self.decimals = self.cfg_sig["decimals"]

        self.delta_r2_str = self.cfg_sig["delta_r2_str"]
        self.t_strng = self.cfg_sig["t_strng"]
        self.p_strng = self.cfg_sig["p_strng"]
        self.p_fdr_strng = self.cfg_sig["p_fdr_strng"]
        self.d_strng = self.cfg_sig["d_strng"]
        self.d_lower_strng = self.cfg_sig["d_lower_strng"]
        self.d_higher_strng = self.cfg_sig["d_higher_strng"]

    def significance_testing(self):
        """
        Conducts significance testing for predictive models and predictor classes.

        This function serves as a wrapper for two key comparisons:
            1. **Model Comparison**:
                - Compares the predictive performance of Elastic Net Regression (ENR) and Random Forest Regression (RFR)
                  across all analyses.
            2. **Predictor Class Comparison**:
                - Evaluates whether adding specific predictor classes to person-level predictors leads to a
                  significant performance increase.

        The function also applies false discovery rate (FDR) correction to control for multiple comparisons
        and generates the final statistical results in terms of t-values and (raw and corrected) p-values.

        ### Model Comparison
            This step compares the predictive performance of the two models, ENR and RFR, across all analyses
            (all, selected, control)
            This results in a total of 24 statistical tests.

        ### Predictor Class Comparison
            This step evaluates the impact of adding various predictor classes on performance. The specific
            comparisons include (always added to pl):
                - Adding `srmc`
                - Adding `sens`
                - Adding `srmc + sens`
                - Adding `mac`
                - Adding `srmc + mac`

            Each comparison is performed under two conditions:
                1. **All Samples**:
                    - Includes all samples
                2. **Selected Samples**:
                    - The reduced datasets to avoid missing data

            These comparisons are carried out separately for both prediction models (ENR and RFR), resulting in
            a total of 20 statistical tests.

        ### Workflow
        1. Model Comparison:
            - Calls `get_model_comparison_data` to retrieve the data.
            - Applies the `apply_compare_models` method for statistical testing.
            - Corrects p-values using FDR correction via `fdr_correct_p_values`.
            - Generates result tables for main and control comparisons using
              `create_sig_results_table_models`.
            - Stores the resulting tables in Excel format if configured.

        2. Predictor Class Comparison:
            - Calls `get_predictor_class_comparison_data` to retrieve the data.
            - Applies the `apply_compare_predictor_classes` method for statistical testing.
            - Corrects p-values using FDR correction via `fdr_correct_p_values`.
            - Generates the result table using `create_sig_results_table_predictor_classes`.
            - Stores the resulting table in Excel format if configured.

        ### File Outputs
        The results are saved as Excel files:
            - Model comparison (main results)
            - Model comparison (control results)
            - Predictor class comparison
        """
        cfg_sig = self.cfg_postprocessing["conduct_significance_tests"]

        file_path_comp_models_main = os.path.join(
            self.sig_result_dir,
            cfg_sig["compare_models"]["filename_compare_models_main"],
        )
        file_path_comp_models_control = os.path.join(
            self.sig_result_dir,
            cfg_sig["compare_models"]["filename_compare_models_control"],
        )
        file_path_comp_predictors = os.path.join(
            self.sig_result_dir,
            cfg_sig["compare_predictor_classes"]["filename_compare_predictor_classes"],
        )

        # compare models
        data_to_compare_models = self.get_model_comparison_data()
        sig_results_models = self.apply_compare_models(data_to_compare_models)
        sig_results_models_fdr = self.fdr_correct_p_values(sig_results_models)
        (
            sig_results_models_main_table,
            sig_results_models_control_table,
        ) = self.create_sig_results_table_models(sig_results_models_fdr)
        if self.cfg_sig["compare_models"]["store"]:
            self.data_saver.save_excel(
                sig_results_models_main_table, file_path_comp_models_main
            )
            self.data_saver.save_excel(
                sig_results_models_control_table, file_path_comp_models_control
            )

        # compare predictor classes
        data_to_compare_predictor_classes = self.get_predictor_class_comparison_data()
        sig_results_predictor_classes = self.apply_compare_predictor_classes(
            data_to_compare_predictor_classes
        )
        sig_results_predictor_classes_fdr = self.fdr_correct_p_values(
            sig_results_predictor_classes
        )
        sig_results_predictor_classes_table = (
            self.create_sig_results_table_predictor_classes(
                sig_results_predictor_classes_fdr
            )
        )

        if self.cfg_sig["compare_predictor_classes"]["store"]:
            self.data_saver.save_excel(
                sig_results_predictor_classes_table, file_path_comp_predictors
            )

    def create_sig_results_table_models(
        self, p_val_dct: NestedDict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates formatted pandas DataFrames from a nested dictionary of statistical results.

        This method processes a nested dictionary structure, extracting statistical results and organizing
        them into two separate DataFrames:
        - A main table (`df_pivoted`) containing results for all predictor classes, except for "control".
        - A separate table (`df_control_pivoted`) specifically for "control" results.

        ### Input Format (`p_val_dct`)
        ```
        {
            "feature_combo": {
                "samples_to_include": {
                    "stat_name": stat_value
                }
            }
        }
        ```

        ### Output
        Two pivoted DataFrames with:
        - **Rows**:
          - Level 1: Sample subsets (e.g., "all", "selected").
          - Level 2: Statistical measures (e.g., `p_val`, `t_val`, `p_val_corrected`).
        - **Columns**: Predictor classes (mapped names based on a configuration).
        - **Values**: Corresponding statistical values for each combination.

        ### Steps
        1. **Flatten the Nested Dictionary**:
           - Iterates through `p_val_dct` and converts its hierarchical structure into a flat list of dictionaries.
        2. **DataFrame Creation**:
           - Creates a DataFrame from the flattened data.
           - Adds custom ordering for predictor classes and statistics based on configuration files.
        3. **Pivot the DataFrame**:
           - Pivots the flat DataFrame to structure the results for easy interpretation.
        4. **Split Tables**:
           - Separates rows related to "control" samples into a separate DataFrame.
           - Drops "control" rows from the main DataFrame.

        Args:
            p_val_dct: A nested dictionary containing statistical results for each feature combination,
                       sample subset, and statistical measure.

        Returns:
            tuple:
                - pd.DataFrame: Main table of results for predictor classes (excluding "control").
                - pd.DataFrame: Separate table of results specifically for "control" samples.
        """
        flattened_data = []

        for feature_combo, inner_dict in p_val_dct.items():
            for samples_to_include, metrics in inner_dict.items():
                for stat, stat_value in metrics.items():
                    flattened_data.append(
                        {
                            "Predictor class": feature_combo,
                            "Samples to include": samples_to_include,
                            "Stat": stat,
                            "Stat value": stat_value,
                        }
                    )

        df = pd.DataFrame(flattened_data)

        custom_order_stats = self.cfg_sig["compare_models"]["stat_order"]
        df["Stat"] = pd.Categorical(
            df["Stat"], categories=custom_order_stats, ordered=True
        )

        custom_order_predictor_classes = list(
            self.feature_combo_name_mapping_main.keys()
        )
        df["Predictor class"] = pd.Categorical(
            df["Predictor class"],
            categories=custom_order_predictor_classes,
            ordered=True,
        )
        df["Predictor class"] = df["Predictor class"].map(
            self.feature_combo_name_mapping_main
        )

        df_pivoted = df.pivot(
            index=["Samples to include", "Stat"],
            columns="Predictor class",
            values="Stat value",
        )

        df_control_pivoted = df_pivoted.loc[("control",), :].copy()
        df_pivoted.drop("control", level="Samples to include", inplace=True)

        return df_pivoted, df_control_pivoted

    def create_sig_results_table_predictor_classes(
        self, p_val_dct: NestedDict
    ) -> pd.DataFrame:
        """
        Creates a pandas DataFrame for predictor classes from a nested dictionary structure.

        This method processes a nested dictionary of statistical results to create a pivoted DataFrame.
        The resulting table organizes results by prediction models, sample subsets, and statistical metrics,
        with columns representing predictor classes.

        ### Input Format (`p_val_dct`)
        ```
        {
            "model_name": {  # e.g., "elasticnet", "randomforestregressor"
                "sample_subset": {  # e.g., "selected", "all"
                    "predictor_class": {  # e.g., "pl", "pl_mac"
                        "stat_name": stat_value
                    }
                }
            }
        }
        ```

        ### Output Format
        A pivoted DataFrame with:
        - **Rows**:
          - Level 1: Prediction model (e.g., `elasticnet`, `randomforestregressor`).
          - Level 2: Sample subset (e.g., `selected`, `all`).
          - Level 3: Statistics (e.g., `p_val`, `t_val`, `p_val_corrected`).
        - **Columns**:
          - Predictor classes, ordered based on a custom mapping.

        ### Steps
        1. **Flatten the Nested Dictionary**:
           - Converts the hierarchical dictionary into a flat list of dictionaries for easier DataFrame creation.
        2. **Create the DataFrame**:
           - Constructs a flat DataFrame from the flattened dictionary.
           - Maps predictor classes and statistics to readable formats using configuration mappings.
           - Orders the columns and rows based on predefined mappings for predictor classes and statistics.
        3. **Pivot the DataFrame**:
           - Organizes the DataFrame into a pivot table with rows and columns matching the desired structure.

        Args:
            p_val_dct: A nested dictionary containing statistical results for each model,
                       sample subset, and predictor class.

        Returns:
            pd.DataFrame: A pivoted DataFrame with multi-index rows (model, sample subset, stat)
                          and columns for predictor classes.
        """
        flattened_data = []

        for model, sample_dict in p_val_dct.items():
            for sample, feature_dict in sample_dict.items():
                for feature, metrics in feature_dict.items():
                    for metric_key, metric_value in metrics.items():
                        flattened_data.append(
                            {
                                "Prediction model": model,
                                "Samples to include": sample,
                                "Predictor class": feature,
                                "Stat": metric_key,
                                "Stat value": metric_value,
                            }
                        )

        df = pd.DataFrame(flattened_data)

        col_order = [
            val
            for key, val in self.cfg_postprocessing["general"]["feature_combinations"][
                "name_mapping"
            ]["main"].items()
            if key in df["Predictor class"].unique()
        ]

        df["Predictor class"] = df["Predictor class"].map(
            self.feature_combo_name_mapping_main
        )
        custom_order = self.cfg_sig["compare_predictor_classes"]["stat_order"]
        df["Stat"] = pd.Categorical(df["Stat"], categories=custom_order, ordered=True)

        df_pivoted = df.pivot(
            index=["Prediction model", "Samples to include", "Stat"],
            columns="Predictor class",
            values="Stat value",
        )
        df_pivoted = df_pivoted[col_order]

        return df_pivoted

    def get_model_comparison_data(self) -> NestedDict:
        """
        Loads and organizes CV results data to prepare for model comparison.

        This method recursively traverses the base results directory to locate JSON files
        containing CV results. It extracts metric values (e.g., "r2") for specific feature
        combinations, sample subsets, and models. The results are structured in a nested
        dictionary for downstream analysis.

        ### Workflow:
        - Traverses directories under `self.sig_result_dir` using `os.walk`.
        - Filters directories based on:
            - The presence of a target feature combination (e.g., "pl", "sens").
            - The sample subset (e.g., "all", "selected").
            - The model name (e.g., "elasticnet", "randomforestregressor").
        - Matches JSON files with the pattern `cv_results_rep_<rep_number>.json`.
        - Reads each matching JSON file and extracts the metric values for the specified feature
          combination, sample subset, and model.
        - The data is stored in a nested `defaultdict` to avoid repeated `setdefault` calls.

        ### Nested Dictionary Structure:
        The method returns a dictionary organized as follows:
        ```
        {
            "feature_combination": {            # e.g., "pl", "sens"
                "sample_subset": {              # e.g., "all", "selected"
                    "model_name": [list of metric values]  # e.g., "elasticnet", "randomforestregressor"
                },
                ...
            },
            ...
        }
        ```

        ### Example:
        ```
        {
            "pl": {
                "all": {
                    "randomforestregressor": [0.8, 0.82, 0.79, ...],  # 500 values
                    "elasticnet": [0.76, 0.78, 0.75, ...]
                },
                "selected": {
                    ...
                },
            },
            ...
        }
        ```

        Returns:
            dict: A dictionary containing CV results for model comparison in the format:
            ```
            {
                "pl": {
                    "all": {
                        "randomforestregressor": [list with 500 values],
                        "elasticnet": [list with 500 values]
                    },
                    "selected": {
                        ...
                    },
                },
                ...
            }
            ```
        """

        feature_combos = list(
            self.cfg_postprocessing["general"]["feature_combinations"]["name_mapping"][
                "main"
            ].keys()
        )
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for dirpath, dirnames, filenames in os.walk(self.sig_result_dir):
            dir_components = os.path.normpath(dirpath).split(os.sep)

            if self.crit not in dir_components:
                continue

            # Check if this directory contains one of the desired feature combos and samples
            feature_combo = next(
                (fc for fc in feature_combos if fc in dir_components), None
            )
            sample = next(
                (s for s in self.samples_to_include if s in dir_components), None
            )
            if not feature_combo or not sample:
                continue

            # Identify the model from the directory path if present
            model = next((m for m in self.models if m in dir_components), None)
            if not model:
                continue

            for filename in filenames:
                if not self.cv_file_matching_pattern.match(filename):
                    continue

                filepath = os.path.join(dirpath, filename)
                data = self.data_loader.read_json(filepath)

                for outer_fold_data in data.values():
                    for metrics in outer_fold_data.values():
                        if self.metric in metrics:
                            results[feature_combo][sample][model].append(
                                metrics[self.metric]
                            )

        results_dct = defaultdict_to_dict(results)

        return results_dct

    def get_predictor_class_comparison_data(self) -> NestedDict:
        """
        Loads and processes cross-validation (CV) results data from JSON files to generate a structured comparison
        of performance metrics across predictor classes (models) and feature combinations.

        This method traverses the base results directory and identifies relevant CV result files based on directory
        structure and file naming patterns. It extracts the performance metrics for specific feature combinations
        and sample types ("all", "selected", and "control") and organizes them into a structured format for
        comparing models.

        The resulting structure (`final_results`) returned by this method is as follows:
        {
            model_name: {
                "all": [
                    {
                        "pl": [...],   # Baseline metrics for "pl" under "all" sample
                        feature_combo: [...]  # Metrics for the current feature combination under "all"
                    },
                    ...
                ],
                "selected": [
                    {
                        "pl": [...],   # Baseline metrics derived from "control" data
                        feature_combo: [...]  # Metrics for the current feature combination under "selected"
                    },
                    ...
                ]
            },
            ...
        }

        Logic Overview:
        - For the "all" scenario:
          * Metrics for "pl" are taken from "all"/"pl" data.
          * Metrics for other feature combinations are paired with "pl" under "all".
        - For the "selected" scenario:
          * Metrics for "pl" are derived from the "control" data for each feature combination.
          * Metrics for other feature combinations are taken from the "selected" data.
        - The method processes all CV result files matching the  given regex pattern from the config  within
          directories that include a known feature combination and sample type.
        - Performance metrics are aggregated based on the specified `metric_to_use` from the configuration.

        Steps Performed:
        1. Traverse the result directory to identify valid paths containing the required feature combinations and samples.
        2. Parse JSON files matching the file naming pattern and extract relevant performance metrics.
        3. Organize the data into a hierarchical structure based on models, sample types, and feature combinations.
        4. Generate metric pairs for comparison between the "pl" baseline and other feature combinations under
           both "all" and "selected" scenarios.

        Returns:
            dict: A structured dictionary containing comparison data for each model, with results categorized
                  under "all" and "selected" scenarios.
        """
        feature_combos = self.cfg_sig["compare_predictor_classes"][
            "feature_combinations_included"
        ]
        predictor_class_ref = self.cfg_sig["compare_predictor_classes"][
            "ref_predictor_class"
        ]

        raw_data = defaultdict(
            lambda: {
                "all": defaultdict(list),
                "selected": defaultdict(list),
                "control": defaultdict(list),
            }
        )

        for dirpath, _, filenames in os.walk(self.sig_result_dir):
            dir_components = os.path.normpath(dirpath).split(os.sep)
            if self.crit not in dir_components:
                continue

            feature_combo = next(
                (fc for fc in feature_combos if fc in dir_components), None
            )
            sample_to_include = next(
                (s for s in self.samples_to_include if s in dir_components), None
            )
            model = next((m for m in self.models if m in dir_components), None)

            for filename in filenames:
                if not self.cv_file_matching_pattern.match(filename):
                    continue
                filepath = os.path.join(dirpath, filename)
                data = self.data_loader.read_json(filepath)

                for _, outer_fold_data in data.items():
                    for imp_data in outer_fold_data.values():
                        if self.metric in imp_data:
                            raw_data[model][sample_to_include][feature_combo].append(
                                imp_data[self.metric]
                            )

        final_results = {}
        for model, model_data in raw_data.items():
            all_pairs = []
            selected_pairs = []
            all_ref = model_data["all"][predictor_class_ref]

            for other in feature_combos:
                if other == predictor_class_ref:
                    continue

                other_all_data = model_data["all"][other]
                all_pairs.append({predictor_class_ref: all_ref, other: other_all_data})
                ref_selected_data = model_data["control"][other]
                other_selected_data = model_data["selected"][other]
                selected_pairs.append(
                    {predictor_class_ref: ref_selected_data, other: other_selected_data}
                )

            final_results[model] = {"all": all_pairs, "selected": selected_pairs}

        return final_results

    def apply_compare_models(self, cv_results_dct: NestedDict) -> NestedDict:
        """
        Applies the corrected dependent t-test to compare the performance of two models
        across feature combinations and sample subsets.

        This method:
        - Iterates through a nested dictionary containing CV results for each model.
        - Extracts metric values (e.g., R²) from the inner dictionary structure.
        - Conducts the corrected dependent t-test between two models (e.g., ElasticNet and RandomForest).
        - Stores the results, including means, standard deviations, t-values, p-values, and delta R², in a dictionary
          that mirrors the original structure.

        ### Dictionary Structure
        Input (`cv_results_dct`):
        ```
        {
            feature_combination: {
                sample_subset: {
                    model_name: [list of CV results for self.metric]
                }
            }
        }
        ```

        Output (`sig_results_dct`):
        ```
        {
            feature_combination: {
                sample_subset: {
                    "M (Model1)": mean_value,
                    "SD (Model1)": sd_value,
                    "M (Model2)": mean_value,
                    "SD (Model2)": sd_value,
                    "delta_R2": difference_between_means,
                    "t": t_value,
                    "p": p_value
                }
            }
        }
        ```

        Args:
            cv_results_dct: A nested dictionary containing CV results for the selected metric (`self.metric`) as a list
                of values. The outer levels represent feature combinations and sample subsets, and the inner level
                contains results for specific models.

        Returns:
            NestedDict: A dictionary with statistical test results replacing the original CV results.
        """
        # sig_results_dct = defaultdict(lambda: defaultdict(dict))
        sig_results_dct = create_defaultdict(n_nesting=2, default_factory=dict)

        for fc, fc_vals in cv_results_dct.items():
            for sti, model_data in fc_vals.items():
                if len(model_data) < 2:
                    print(f"WARNING: Not enough models to compare in {fc}/{sti}, SKIP")
                    continue

                model1_name, model2_name = list(model_data.keys())
                cv_results_model1, cv_results_model2 = (
                    model_data[model1_name],
                    model_data[model2_name],
                )

                mean1, sd1 = np.mean(cv_results_model1), np.std(cv_results_model1)
                mean2, sd2 = np.mean(cv_results_model2), np.std(cv_results_model2)

                t_val, p_val = self.corrected_dependent_ttest(
                    cv_results_model1, cv_results_model2
                )
                d, d_lower, d_upper = self.compute_effect_size_CI(
                    t_val=t_val,
                    n=500,
                    alpha=0.05,
                    round_digits=2
                )
                delta_R2 = np.round(
                    np.round(mean2, self.decimals) - np.round(mean1, self.decimals),
                    self.decimals,
                )

                sig_results_dct[fc][sti] = {
                    f"M (SD) {self.model_name_mapping[model1_name]}": f"{mean1:.{self.decimals}f} ({sd1:.{self.decimals}f})",
                    f"M (SD) {self.model_name_mapping[model2_name]}": f"{mean2:.{self.decimals}f} ({sd2:.{self.decimals}f})",
                    self.delta_r2_str: f"{delta_R2:.{self.decimals}f}",
                    self.t_strng: f"{t_val:.{self.decimals}f}",
                    self.p_strng: p_val,  # Kept as is, assuming p-value formatting is handled elsewhere
                    self.d_strng: d,
                    self.d_lower_strng: d_lower,
                    self.d_higher_strng: d_upper
                }

        return defaultdict_to_dict(sig_results_dct)

    def apply_compare_predictor_classes(self, cv_results_dct: NestedDict) -> NestedDict:
        """
        Compares the performance of predictor classes using the corrected dependent t-test.

        This method:
        - Iterates through cross-validation (CV) results for each model and sample subset.
        - Performs a corrected dependent t-test to compare performance between:
            - The person-level predictors (`pl`).
            - A combination of person-level predictors with another predictor class (e.g., `pl+srmc`).
        - Computes descriptive statistics (mean and standard deviation) for both groups.
        - Stores results (e.g., mean, SD, t-value, p-value, delta R²) in a nested dictionary
          that mirrors the structure of the input dictionary.

        ### Input Format (`cv_results_dct`)
        ```
        {
            "model_name": {  # e.g., "elasticnet", "randomforestregressor"
                "sample_subset": {  # e.g., "all", "selected"
                    "comparison_key": {  # e.g., {"pl": [...], "pl+class": [...]}
                        ...
                    }
                }
            }
        }
        ```

        ### Output Format
        ```
        {
            "model_name": {
                "sample_subset": {
                    "comparison_key": {
                        "M (SD) Personal": "mean (SD) of pl values",
                        "M (SD) Other": "mean (SD) of pl+class values",
                        "deltaR2": "difference between means",
                        "p": "p-value from t-test",
                        "t": "t-value from t-test"
                    }
                }
            }
        }
        ```

        Args:
            cv_results_dct: Nested dictionary containing CV results for person-level predictors (`pl`)
                            and their combinations with other predictor classes.

        Returns:
            NestedDict: A nested dictionary containing statistical test results,
                        including means, SDs, t-values, p-values, and delta R².
        """
        sig_results_dct = defaultdict(lambda: defaultdict(dict))

        for model, model_vals in cv_results_dct.items():
            for samples_to_include, comparisons in model_vals.items():
                for comparison in comparisons:
                    pl_data, pl_combo_data = (
                        comparison.get("pl"),
                        list(comparison.values())[1],
                    )

                    if pl_data and pl_combo_data:
                        mean_pl, sd_pl = np.mean(pl_data), np.std(pl_data)
                        mean_combo, sd_combo = np.mean(pl_combo_data), np.std(
                            pl_combo_data
                        )
                        t_val, p_val = self.corrected_dependent_ttest(
                            pl_combo_data, pl_data
                        )
                        d, d_lower, d_upper = self.compute_effect_size_CI(
                            t_val=t_val,
                            n=500,
                            alpha=0.05,
                            round_digits=2
                        )
                        delta_R2 = np.round(
                            np.round(mean_combo, self.decimals)
                            - np.round(mean_pl, self.decimals),
                            self.decimals,
                        )

                        sig_results_dct[model][samples_to_include][
                            list(comparison.keys())[1]
                        ] = {
                            f"M (SD) Personal": f"{mean_pl:.{self.decimals}f} ({sd_pl:.{self.decimals}f})",
                            f"M (SD) Other": f"{mean_combo:.{self.decimals}f} ({sd_combo:.{self.decimals}f})",
                            self.delta_r2_str: f"{delta_R2:.{self.decimals}f}",
                            self.p_strng: p_val,  # Kept as is, assuming p-value formatting is handled elsewhere
                            self.t_strng: f"{t_val:.{self.decimals}f}",
                            self.d_strng: d,
                            self.d_lower_strng: d_lower,
                            self.d_higher_strng: d_upper
                        }

        return defaultdict_to_dict(sig_results_dct)

    def fdr_correct_p_values(self, p_val_dct: NestedDict) -> NestedDict:
        """
        Correct p-values using False Discovery Rate (FDR) as described by Benjamini & Hochberg (1995).
        This function
            - recursively finds all instances of 'p_val' in a nested dictionary structure
            - apply the FDR correction
            - formats the p-values if a formatting function is provided
            - recreates the dict structure of the input and adds the FDR corrected p-values

        Args:
            p_val_dct: Dict containing the t-values and the uncorrected p-values for each comparison to conduct.
                The outer dict hierarchy depends on the type of comparison (models vs. predictor classes).

        Returns:
            result_dict: Dict, with 'p (FDR corrected)' values added in the inner Dict.
        """
        p_val_fdr_dct = copy.deepcopy(p_val_dct)
        p_values = []
        p_val_locations = []

        def collect_p_values(d: dict[str, Any], p_strng: str) -> None:
            """
            Recursively collect all 'p_val' values and their containing dictionaries.

            Args:
                d: The Dict to traverse.
            """
            for key, value in d.items():
                if isinstance(value, dict):
                    collect_p_values(value, p_strng)
                elif key == p_strng:
                    p_values.append(value)
                    p_val_locations.append(d)

        collect_p_values(p_val_fdr_dct, self.p_strng)

        if p_values:
            _, adjusted_p_values = fdrcorrection(p_values)

            formatted_p_values_fdr = format_p_values(adjusted_p_values)
            formatted_p_values = format_p_values(p_values)

            for loc, adj_p, orig_p in zip(
                p_val_locations, formatted_p_values_fdr, formatted_p_values
            ):
                loc[self.p_fdr_strng] = adj_p
                loc[self.p_strng] = orig_p

        return p_val_fdr_dct

    @staticmethod
    def corrected_dependent_ttest(
        data1: list[float], data2: list[float], test_training_ratio: float = 1 / 9
    ):
        """
        Python implementation for the corrected paired t-test as described by Nadeau & Bengio (2003) and
        Bouckaert & Frank (2004).
        See also https://gist.github.com/jensdebruijn/13e8eeda85eb8644ac2a4ac4c3b8e732

        Args:
            data1: list, containing the prediction results for a certain setting (up to a specific model)
            data2: list, containing the prediction results for a another setting (up to a specific model)
            test_training_ratio: float, depends on the number of folds in the outer_cv (i.e., 10 in this setting).
                So 1/10 of the data is used for testing, 9/10 for training.

        Returns:
            t_stat: float, t statistic of the comparison of data1 and data2
            p: float, p-value for the comparison of data1 and data2
        """
        n = len(data1)
        differences = [(data1[i] - data2[i]) for i in range(n)]
        sd = stdev(differences)
        divisor = 1 / n * sum(differences)
        denominator = sqrt(1 / n + test_training_ratio) * sd

        t_stat = divisor / denominator
        df = n - 1  # degrees of freedom
        p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0

        return t_stat, p

    @staticmethod
    def compute_effect_size_CI(
        t_val: float, n: int = 500, alpha: float = 0.05, round_digits: int = 2
    ) -> tuple[float, float, float]:
        """
        Computes Cohen's d and its confidence interval using a large-sample approximation.

        - Calculates Cohen's dz directly from the t-statistic and sample size.
        - Estimates the standard error of dz and applies the t-distribution critical value.
        - Returns the rounded effect size along with the rounded confidence bounds.

        Args:
            t_val: t-statistic value from a paired t-test.
            n: Number of pairs in the dataset (i.e., number of outer folds). Defaults to 500.
            alpha: Significance level for the CI. Defaults to 0.05 (95% CI).
            round_digits: Number of decimal places for rounding. Defaults to 2.

        Returns:
            Tuple[float, float, float]: Rounded Cohen's dz, lower CI bound, upper CI bound.
        """
        d = t_val / np.sqrt(n)
        se_d = np.sqrt(1 / n + (d ** 2) / (2 * n))
        t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)

        lower = d - t_crit * se_d
        upper = d + t_crit * se_d

        return round(d, round_digits), round(lower, round_digits), round(upper, round_digits)
